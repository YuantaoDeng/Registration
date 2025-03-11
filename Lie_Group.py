import SimpleITK as sitk
import numpy as np
from scipy.ndimage import gaussian_filter

class CustomDemonsRegistrationFilter:
    def __init__(self):
        # 默认参数设置
        self.number_of_iterations = 20
        self.maximum_step_length = 2.0
        self.exp_steps = 4  # scaling-and-squaring steps
        # symmetric demons force
        # diffeomorphic demons type
        self.use_regularization = True
        self.smoothing_sigma = 1.0

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

    def Execute(self, fixed_image, moving_image, displacement_field):
        """
        conducting demons registration, using symmetric demons force and diffeomorphic demons type.
        parameters：
          - fixed_image
          - moving_image
          - displacement_field: SimpleITK displacement image (VectorFloat)
        """
        fixed_arr = sitk.GetArrayFromImage(fixed_image).astype(np.float32)
        moving_arr = sitk.GetArrayFromImage(moving_image).astype(np.float32)
        disp_arr = sitk.GetArrayFromImage(displacement_field).astype(np.float64)


        for it in range(self.number_of_iterations):
            # 1. resample using displacement field
            disp_img = sitk.GetImageFromArray(disp_arr, isVector=True)
            disp_img.CopyInformation(fixed_image)
            transform = sitk.DisplacementFieldTransform(disp_img)
            warped_moving = sitk.Resample(moving_image, fixed_image, transform, sitk.sitkLinear, 0.0)
            warped_moving_arr = sitk.GetArrayFromImage(warped_moving).astype(np.float32)

            # 2. calculate intensity diff
            diff = fixed_arr - warped_moving_arr  # shape: [z, y, x]

            # 3. calculate the gradient (symmetric force)
            grad_fixed = self.compute_gradient(fixed_arr)         # shape: [z, y, x, 3]
            grad_warped = self.compute_gradient(warped_moving_arr)  # shape: [z, y, x, 3]
            grad = -0.5 * (grad_fixed + grad_warped)

            # 4. calculate update vector for every pixel: update = - (diff * grad) / (||grad||^2 + diff^2)
            grad_norm_sq = np.sum(grad**2, axis=-1, keepdims=True)  # shape: [z, y, x, 1]
            denominator = grad_norm_sq + np.square(diff)[..., np.newaxis] + 1e-8  # incase 0
            update = - (diff[..., np.newaxis] * grad) / denominator

            # 5. normalization
            update_norm = np.sqrt(np.sum(update**2, axis=-1))
            max_update = np.max(update_norm)
            if max_update > 0:
                update = update / max_update * self.maximum_step_length

            # 6. gaussian smoothing
            if self.use_regularization:
                update = self.gaussian_smooth(update, sigma=self.smoothing_sigma)

            # 7. diffeomorphic update
            # updating using exp function (scaling-and-squaring)
            exp_update = self.exponential_map(update, self.exp_steps)
            exp_update_img = sitk.GetImageFromArray(exp_update, isVector=True)
            exp_update_img.CopyInformation(fixed_image)
            exp_transform = sitk.DisplacementFieldTransform(exp_update_img)
            disp_img = sitk.GetImageFromArray(disp_arr, isVector=True)
            disp_img.CopyInformation(fixed_image)
            warped_disp = sitk.Resample(disp_img, fixed_image, exp_transform, sitk.sitkLinear, 0.0)
            warped_disp_arr = sitk.GetArrayFromImage(warped_disp)
            disp_arr = warped_disp_arr + exp_update

        # return to SimpleITK image (VectorFloat64)
        result_img = sitk.GetImageFromArray(disp_arr, isVector=True)
        result_img.CopyInformation(fixed_image)
        result_img = sitk.Cast(result_img, sitk.sitkVectorFloat64)
        return result_img

    def compute_gradient(self, image_array):
        gradients = np.gradient(image_array)
        grad_stack = np.stack(gradients, axis=-1)
        return grad_stack

    def gaussian_smooth(self, vector_field, sigma):
        smoothed = np.empty_like(vector_field)
        for d in range(vector_field.shape[-1]):
            smoothed[..., d] = gaussian_filter(vector_field[..., d], sigma=sigma)
        return smoothed

    def exponential_map(self, vector_field, exp_steps):
        """
        parameters：
          - vector_field:
          - exp_steps:
        return approximation of exp(vector_field)
        """
        scaled_field = vector_field / (2 ** exp_steps)
        result = scaled_field.copy()
        for _ in range(exp_steps):
            # result = result + warp(result, result)
            result = result + self.warp_vector_field(result, result)
        return result

    def warp_vector_field(self, field, warp_field):
        field = field.astype(np.float64)
        warp_field = warp_field.astype(np.float64)
        field_img = sitk.GetImageFromArray(field, isVector=True)
        warp_img = sitk.GetImageFromArray(warp_field, isVector=True)
        warp_img.CopyInformation(field_img)
        warp_img = sitk.Cast(warp_img, sitk.sitkVectorFloat64)
        transform = sitk.DisplacementFieldTransform(warp_img)
        resampled = sitk.Resample(field_img, field_img, transform, sitk.sitkLinear, 0.0)
        resampled_arr = sitk.GetArrayFromImage(resampled)
        return resampled_arr
