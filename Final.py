import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

from Sitk2 import multiscale_demons_2d
from Lie_Group import CustomDemonsRegistrationFilter

# read initial image
fixed_path = r"C:\Users\Kaze\Desktop\Tyche\Tyche-A7_0.png"
moving_path = r"C:\Users\Kaze\Desktop\Tyche\Tyche-A7_1.png"
fixed_image = sitk.ReadImage(fixed_path)
moving_image = sitk.ReadImage(moving_path)

def rgb_to_gray(image):
    red = sitk.VectorIndexSelectionCast(image, 0)
    green = sitk.VectorIndexSelectionCast(image, 1)
    blue = sitk.VectorIndexSelectionCast(image, 2)

    red = sitk.Cast(red, sitk.sitkFloat32)
    green = sitk.Cast(green, sitk.sitkFloat32)
    blue = sitk.Cast(blue, sitk.sitkFloat32)

    gray = 0.2989 * red + 0.5870 * green + 0.1140 * blue  # calculating by simple summation
    return gray


fixed_gray = rgb_to_gray(fixed_image)
moving_gray = rgb_to_gray(moving_image)


# whether to use custom algorithms
use_custom = False

if use_custom:
    print("Using CustomDemonsRegistrationFilter...")
    reg_filter = CustomDemonsRegistrationFilter()
    reg_filter.SetNumberOfIterations(20)
    reg_filter.SetMaximumStepLength(2.0)
    reg_filter.SetExponentialSteps(1)
    reg_filter.SetSmoothDisplacementField(True)
    reg_filter.SetStandardDeviations(1.0)
    registration_algorithm = reg_filter
else:
    print("Using SimpleITK DiffeomorphicDemonsRegistrationFilter...")
    registration_algorithm = sitk.SymmetricForcesDemonsRegistrationFilter()
    registration_algorithm.SetNumberOfIterations(20)
    registration_algorithm.SetSmoothDisplacementField(True)
    registration_algorithm.SetStandardDeviations(1.0)

# conducting registration
tx = multiscale_demons_2d(
    registration_algorithm=registration_algorithm,
    fixed_image=fixed_gray,
    moving_image=moving_gray,
    shrink_factors=[4, 2],
    smoothing_sigmas=[8.0, 4.0],
)


# resample using the transform we get
registered_image = sitk.Resample(moving_gray, fixed_gray, tx, sitk.sitkBSpline, 0.0)

# plotting
fixed_arr = sitk.GetArrayFromImage(fixed_gray)
moving_arr = sitk.GetArrayFromImage(moving_gray)
registered_arr = sitk.GetArrayFromImage(registered_image)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(fixed_arr, cmap="gray")
axes[0].set_title("Fixed Gray Image")
axes[0].axis("off")

axes[1].imshow(moving_arr, cmap="gray")
axes[1].set_title("Moving Gray Image")
axes[1].axis("off")

axes[2].imshow(registered_arr, cmap="gray")
axes[2].set_title("Registered Image")
axes[2].axis("off")

plt.tight_layout()
plt.show()
