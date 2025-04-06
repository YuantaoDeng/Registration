import SimpleITK as sitk
import numpy as np
import torch
import torch.nn.functional as F

def gaussian_kernel2d(kernel_size, sigma, device, dtype):
    # Construct a 2D Gaussian kernel
    ax = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2.0
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel = kernel / torch.sum(kernel)
    return kernel

class CustomDemonsRegistrationFilter:
    def __init__(self):
        self.number_of_iterations = 20
        self.maximum_step_length = 2.0
        self.exp_steps = 4  # scaling-and-squaring steps
        self.use_regularization = True
        self.smoothing_sigma = 1.0
        # For iteration callback support
        self.iteration_callbacks = []
        self.elapsed_iterations = 0
        self.metric = 0.0

    def SetNumberOfIterations(self, n):
        self.number_of_iterations = n

    def SetMaximumStepLength(self, length):
        self.maximum_step_length = length

    def SetExponentialSteps(self, steps):
        self.exp_steps = steps

    def SetSmoothDisplacementField(self, flag):
        self.use_regularization = flag

    def SetStandardDeviations(self, sigma):
        self.smoothing_sigma = sigma

    def AddCommand(self, event, callback):
        # Add a callback function for a given event (only iteration event is supported in this implementation)
        # The 'event' parameter is ignored here.
        self.iteration_callbacks.append(callback)

    def GetElapsedIterations(self):
        return self.elapsed_iterations

    def GetMetric(self):
        return self.metric

    def Execute(self, fixed_image, moving_image, displacement_field):
        """
        Conducts demons registration using symmetric demons force and a diffeomorphic update.
        Parameters:
          - fixed_image: SimpleITK image (grayscale)
          - moving_image: SimpleITK image (grayscale)
          - displacement_field: SimpleITK displacement field image (VectorFloat)
        Returns:
          - Registered displacement field as a SimpleITK VectorFloat64 image.
        """
        # Convert SimpleITK images to numpy arrays and then to torch tensors (assumes 2D images)
        fixed_arr = sitk.GetArrayFromImage(fixed_image).astype(np.float32)
        moving_arr = sitk.GetArrayFromImage(moving_image).astype(np.float32)
        disp_arr = sitk.GetArrayFromImage(displacement_field).astype(np.float32)

        # Assume fixed/moving images are [H, W] and displacement field is [H, W, 2]
        fixed_tensor = torch.from_numpy(fixed_arr)
        moving_tensor = torch.from_numpy(moving_arr)
        disp_tensor = torch.from_numpy(disp_arr)

        # Set device (CPU in this example)
        device = torch.device("cpu")
        fixed_tensor = fixed_tensor.to(device)
        moving_tensor = moving_tensor.to(device)
        disp_tensor = disp_tensor.to(device)

        for it in range(self.number_of_iterations):
            # 1. Resample the moving image using the current displacement field
            warped_moving = self.warp_image_torch(moving_tensor, disp_tensor)
            # 2. Compute the intensity difference between fixed and warped moving images
            diff = fixed_tensor - warped_moving  # shape: [H, W]
            # 3. Compute gradients of the fixed and warped moving images (symmetric force)
            grad_fixed = self.compute_gradient_torch(fixed_tensor)   # shape: [H, W, 2]
            grad_warped = self.compute_gradient_torch(warped_moving)   # shape: [H, W, 2]
            grad = -0.5 * (grad_fixed + grad_warped)
            # 4. Compute update vector for every pixel: update = - (diff * grad) / (||grad||^2 + diff^2)
            diff_unsq = diff.unsqueeze(-1)  # shape: [H, W, 1]
            grad_norm_sq = torch.sum(grad ** 2, dim=-1, keepdim=True)  # shape: [H, W, 1]
            denominator = grad_norm_sq + diff_unsq ** 2 + 1e-8  # prevent division by zero
            update = - (diff_unsq * grad) / denominator  # shape: [H, W, 2]
            # 5. Normalize the update vector
            update_norm = torch.sqrt(torch.sum(update ** 2, dim=-1))
            max_update = update_norm.max()
            if max_update > 0:
                update = update / max_update * self.maximum_step_length
            # 6. Apply Gaussian smoothing (regularization)
            if self.use_regularization:
                update = self.gaussian_smooth_torch(update, sigma=self.smoothing_sigma)
            # 7. Diffeomorphic update using scaling-and-squaring to compute the exponential map
            exp_update = self.exponential_map(update, self.exp_steps)
            # Warp the current displacement field using the exponential update and add the update
            warped_disp = self.warp_vector_field_torch(disp_tensor, exp_update)
            disp_tensor = warped_disp + exp_update

            # Update the iteration count and metric (mean squared intensity difference)
            self.elapsed_iterations = it + 1
            self.metric = torch.mean(diff ** 2).item()

            # Call iteration callbacks if any are registered
            for callback in self.iteration_callbacks:
                callback(self)

        # Convert the resulting displacement field back to a SimpleITK image (VectorFloat64)
        disp_np = disp_tensor.cpu().numpy()
        result_img = sitk.GetImageFromArray(disp_np, isVector=True)
        result_img.CopyInformation(fixed_image)
        result_img = sitk.Cast(result_img, sitk.sitkVectorFloat64)
        return result_img

    def compute_gradient_torch(self, image):
        """
        Compute the gradient using central differences, assuming image shape [H, W].
        Returns a tensor of shape [H, W, 2] where the first component is the gradient in y and the second in x.
        """
        image = image.unsqueeze(0).unsqueeze(0)  # shape: (1, 1, H, W)
        kernel_x = torch.tensor([[-0.5, 0, 0.5]], dtype=image.dtype, device=image.device).view(1, 1, 1, 3)
        kernel_y = torch.tensor([[-0.5], [0], [0.5]], dtype=image.dtype, device=image.device).view(1, 1, 3, 1)
        grad_x = F.conv2d(image, kernel_x, padding=(0, 1))
        grad_y = F.conv2d(image, kernel_y, padding=(1, 0))
        grad_x = grad_x.squeeze(0).squeeze(0)
        grad_y = grad_y.squeeze(0).squeeze(0)
        # Stack gradients in the order: [gradient_y, gradient_x]
        grad = torch.stack([grad_y, grad_x], dim=-1)
        return grad

    def gaussian_smooth_torch(self, vector_field, sigma):
        """
        Apply Gaussian smoothing to the vector_field (shape: [H, W, C]) using conv2d on each channel.
        """
        H, W, C = vector_field.shape
        vector_field_t = vector_field.permute(2, 0, 1).unsqueeze(0)  # shape: [1, C, H, W]
        kernel_size = int(2 * (3 * sigma) + 1)
        kernel = gaussian_kernel2d(kernel_size, sigma, device=vector_field.device, dtype=vector_field.dtype)
        # Expand kernel to each channel: shape [C, 1, kernel_size, kernel_size]
        kernel = kernel.expand(C, 1, kernel_size, kernel_size)
        padding = kernel_size // 2
        smoothed = F.conv2d(vector_field_t, kernel, padding=padding, groups=C)
        smoothed = smoothed.squeeze(0).permute(1, 2, 0)  # shape: [H, W, C]
        return smoothed

    def exponential_map(self, vector_field, exp_steps):
        """
        Compute an approximation of the exponential map of vector_field using scaling-and-squaring.
        Parameters:
          - vector_field: tensor of shape [H, W, C]
          - exp_steps: number of squaring steps
        Returns:
          - An approximation of exp(vector_field)
        """
        scaled_field = vector_field / (2 ** exp_steps)
        result = scaled_field.clone()
        for _ in range(exp_steps):
            result = result + self.warp_vector_field_torch(result, result)
        return result

    def warp_vector_field_torch(self, field, warp_field):
        """
        Warp the vector field 'field' using the warp_field via grid_sample.
        Assumes both field and warp_field are of shape [H, W, 2] (for the 2D case).
        """
        H, W, _ = field.shape
        dtype = field.dtype
        device = field.device
        ys = torch.linspace(0, H - 1, H, device=device, dtype=dtype)
        xs = torch.linspace(0, W - 1, W, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        base_grid = torch.stack((grid_x, grid_y), dim=-1)  # shape: [H, W, 2]
        sampling_coords = base_grid + warp_field  # pixel coordinates
        # Normalize coordinates to [-1, 1]
        sampling_coords_x = 2.0 * sampling_coords[..., 0] / (W - 1) - 1.0
        sampling_coords_y = 2.0 * sampling_coords[..., 1] / (H - 1) - 1.0
        normalized_grid = torch.stack((sampling_coords_x, sampling_coords_y), dim=-1)
        normalized_grid = normalized_grid.unsqueeze(0)  # shape: [1, H, W, 2]
        # Convert field to shape [1, C, H, W]
        field_t = field.permute(2, 0, 1).unsqueeze(0)
        warped_field = F.grid_sample(field_t, normalized_grid, mode='bilinear', align_corners=True, padding_mode='zeros')
        warped_field = warped_field.squeeze(0).permute(1, 2, 0)
        return warped_field

    def warp_image_torch(self, image, disp):
        """
        Warp a grayscale image (shape: [H, W]) using the displacement field 'disp'
        and return the warped image.
        """
        H, W = image.shape
        dtype = image.dtype
        device = image.device
        ys = torch.linspace(0, H - 1, H, device=device, dtype=dtype)
        xs = torch.linspace(0, W - 1, W, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        base_grid = torch.stack((grid_x, grid_y), dim=-1)  # shape: [H, W, 2]
        sampling_coords = base_grid + disp  # disp: shape: [H, W, 2]
        sampling_coords_x = 2.0 * sampling_coords[..., 0] / (W - 1) - 1.0
        sampling_coords_y = 2.0 * sampling_coords[..., 1] / (H - 1) - 1.0
        normalized_grid = torch.stack((sampling_coords_x, sampling_coords_y), dim=-1)
        normalized_grid = normalized_grid.unsqueeze(0)  # shape: [1, H, W, 2]
        image_t = image.unsqueeze(0).unsqueeze(0)  # shape: [1, 1, H, W]
        warped_image = F.grid_sample(image_t, normalized_grid, mode='bilinear', align_corners=True, padding_mode='zeros')
        return warped_image.squeeze(0).squeeze(0)

# Example iteration callback function
def iteration_callback(filter):
    print("\rIteration {0}: Metric = {1:.2f}".format(filter.GetElapsedIterations(), filter.GetMetric()), end="")
