import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import pirt
from skimage.metrics import structural_similarity

# 配准参数设置
registration_config = {
    'img_type': "enhanced_img",
    'grid_sampling_factor': 1,
    'scale_sampling': 20,
    'speed_factor': 3
}


def rgb_to_gray(image):
    red = sitk.VectorIndexSelectionCast(image, 0)
    green = sitk.VectorIndexSelectionCast(image, 1)
    blue = sitk.VectorIndexSelectionCast(image, 2)
    red = sitk.Cast(red, sitk.sitkFloat32)
    green = sitk.Cast(green, sitk.sitkFloat32)
    blue = sitk.Cast(blue, sitk.sitkFloat32)
    gray = 0.2989 * red + 0.5870 * green + 0.1140 * blue
    return gray



fixed_path = r"Tyche-A7_0.png"
moving_path = r"Tyche-A7_1.png"


fixed_img = sitk.ReadImage(fixed_path)
moving_img = sitk.ReadImage(moving_path)


if fixed_img.GetNumberOfComponentsPerPixel() > 1:
    fixed_gray = rgb_to_gray(fixed_img)
else:
    fixed_gray = sitk.Cast(fixed_img, sitk.sitkFloat32)

if moving_img.GetNumberOfComponentsPerPixel() > 1:
    moving_gray = rgb_to_gray(moving_img)
else:
    moving_gray = sitk.Cast(moving_img, sitk.sitkFloat32)


#Only choose the common area
def crop_images_to_common_region(images):
    arrays = [sitk.GetArrayFromImage(img) for img in images]
    common_height = min(arr.shape[0] for arr in arrays)
    common_width = min(arr.shape[1] for arr in arrays)
    print("Common region size:", (common_height, common_width))

    cropped_images = []
    for img, arr in zip(images, arrays):
        cropped_arr = arr[:common_height, :common_width]
        cropped_img = sitk.GetImageFromArray(cropped_arr)
        cropped_img.SetSpacing(img.GetSpacing())
        cropped_img.SetDirection(img.GetDirection())
        cropped_img.SetOrigin(img.GetOrigin())
        cropped_images.append(cropped_img)
    return cropped_images


cropped_gray_images = crop_images_to_common_region([fixed_gray, moving_gray])
fixed_gray_cropped = cropped_gray_images[0]
moving_gray_cropped = cropped_gray_images[1]

# Normalization
fixed_np = sitk.GetArrayFromImage(fixed_gray_cropped).astype(np.float32) / 255.0
moving_np = sitk.GetArrayFromImage(moving_gray_cropped).astype(np.float32) / 255.0


# Construct pirt Aarray
fixed_arr = pirt.Aarray(fixed_np, (0.6, 2.0))
moving_arr = pirt.Aarray(moving_np, (0.6, 2.0))

# ----------------------------
# Step 5: 对 moving 图像进行配准，目标是匹配 fixed 图像
# ----------------------------
print("Registering moving image to fixed image...")
reg = pirt.DiffeomorphicDemonsRegistration(fixed_arr, moving_arr)
reg.params.deform_wise = 'pairwise'
reg.params.grid_sampling_factor = registration_config['grid_sampling_factor']
reg.params.scale_sampling = registration_config['scale_sampling']
reg.params.speed_factor = registration_config['speed_factor']
reg.register()
reg_result = reg.get_final_deform(0, 1).apply_deformation(moving_arr)
registered_np = np.array(reg_result)

#Evaluation
diff = np.abs(fixed_np - registered_np)
ncc = np.corrcoef(fixed_np.flatten(), registered_np.flatten())[0, 1]
ssim_val = structural_similarity(fixed_np, registered_np, data_range=registered_np.max() - registered_np.min())
print(f"Normalized Cross-Correlation (NCC): {ncc:.4f}")
print(f"Structural Similarity Index (SSIM): {ssim_val:.4f}")

#Plotting
import matplotlib.pyplot as plt

plt.figure(figsize=(6.4, 4.8), dpi=100)
plt.imshow(fixed_np, cmap='gray', interpolation="none")
plt.title("PIRT (Fixed)")
plt.axis("off")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6.4, 4.8), dpi=100)
plt.imshow(moving_np, cmap='gray', interpolation="none")
plt.title("PIRT (Moving)")
plt.axis("off")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6.4, 4.8), dpi=100)
plt.imshow(registered_np, cmap='gray', interpolation="none")
plt.title("PIRT (Registered)")
plt.axis("off")
plt.tight_layout()
plt.show()

images = [fixed_np, moving_np, registered_np]
titles = ["Fixed Image", "Moving Image", "Registered Image"]


fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i in range(3):
    axes[i].imshow(images[i], cmap='gray', interpolation="none")
    axes[i].set_title(titles[i])
    axes[i].axis('off')

fig.suptitle("PIRT", fontsize=16, y=0.95)

plt.tight_layout()
plt.show()



