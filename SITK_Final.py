import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from Torch_based import CustomDemonsRegistrationFilter


# read initial image
fixed_path = r"Tyche-A7_0.png"
moving_path = r"Tyche-A7_1.png"
fixed_image = sitk.ReadImage(fixed_path)
moving_image = sitk.ReadImage(moving_path)

def rgb_to_gray(image):
    red = sitk.VectorIndexSelectionCast(image, 0)
    green = sitk.VectorIndexSelectionCast(image, 1)
    blue = sitk.VectorIndexSelectionCast(image, 2)

    red = sitk.Cast(red, sitk.sitkFloat64)
    green = sitk.Cast(green, sitk.sitkFloat64)
    blue = sitk.Cast(blue, sitk.sitkFloat64)

    gray = 0.2989 * red + 0.5870 * green + 0.1140 * blue  # calculating by simple summation
    return gray


def smooth_and_resample(image, shrink_factor, smoothing_sigma):
    """
    Args:
        image: waiting for resample
        shrink_factor: new size = origin size/shrink_factor。
        smoothing_sigma: std of gaussian smoothing
    Return:
        image after smoothing and resampling
    """
    smoothed_image = sitk.SmoothingRecursiveGaussian(image, smoothing_sigma)

    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_size = [int(sz / float(shrink_factor) + 0.5) for sz in original_size]
    new_spacing = [
        ((original_sz - 1) * original_spc) / (new_sz - 1)
        for original_sz, original_spc, new_sz in zip(original_size, original_spacing, new_size)
    ]
    return sitk.Resample(
        smoothed_image,
        new_size,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,  # could change to other methods
        image.GetOrigin(),
        new_spacing,
        image.GetDirection(),
        0.0,
        image.GetPixelID(),
    )


def multiscale_demons_2d(
        registration_algorithm,
        fixed_image,
        moving_image,
        initial_transform=None,
        shrink_factors=None,
        smoothing_sigmas=None,
):
    """
    Args:
        registration_algorithm: Execute(fixed_image, moving_image, displacement_field_image)
        fixed_image
        moving_image
        initial_transform: sitk transform to initialize the displacement field
        shrink_factors: list of shrink factors
        smoothing_sigmas: smoothing sigma under every scale
    Returns:
        displacement field in SimpleITK.DisplacementFieldTransform
    """
    fixed_images = [fixed_image]
    moving_images = [moving_image]
    if shrink_factors is not None:
        for shrink_factor, smoothing_sigma in reversed(list(zip(shrink_factors, smoothing_sigmas))):
            fixed_images.append(smooth_and_resample(fixed_images[0], shrink_factor, smoothing_sigma))
            moving_images.append(smooth_and_resample(moving_images[0], shrink_factor, smoothing_sigma))

    # initialize the displacement field
    if initial_transform:
        initial_displacement_field = sitk.TransformToDisplacementField(
            initial_transform,
            sitk.sitkVectorFloat64,
            fixed_images[-1].GetSize(),
            fixed_images[-1].GetOrigin(),
            fixed_images[-1].GetSpacing(),
            fixed_images[-1].GetDirection(),
        )
    else:
        initial_displacement_field = sitk.Image(
            [fixed_images[-1].GetWidth(), fixed_images[-1].GetHeight()],
            sitk.sitkVectorFloat64,
            2
        )
        initial_displacement_field.CopyInformation(fixed_images[-1])

    # conduct registration in the lowest resolution
    initial_displacement_field = registration_algorithm.Execute(
        fixed_images[-1], moving_images[-1], initial_displacement_field
    )

    for f_image, m_image in reversed(list(zip(fixed_images[:-1], moving_images[:-1]))):
        initial_displacement_field = sitk.Resample(initial_displacement_field, f_image)
        initial_displacement_field = registration_algorithm.Execute(
            f_image, m_image, initial_displacement_field
        )

    return sitk.DisplacementFieldTransform(initial_displacement_field)


def iteration_callback(filter):
    print("\r{0}: {1:.2f}".format(filter.GetElapsedIterations(), filter.GetMetric()), end="")


fixed_gray = rgb_to_gray(fixed_image)
moving_gray = rgb_to_gray(moving_image)


registration_algorithm = sitk.DiffeomorphicDemonsRegistrationFilter()
registration_algorithm.SetMaximumUpdateStepLength(1.0)
registration_algorithm.SetNumberOfIterations(200)
registration_algorithm.SetSmoothDisplacementField(True)
registration_algorithm.SetStandardDeviations(1.0)
registration_algorithm.AddCommand(sitk.sitkIterationEvent, lambda: iteration_callback(registration_algorithm))


shrink_factors = [4, 2, 1]
smoothing_sigmas = [1.5, 0.75, 0.25]


# conducting registration
tx = multiscale_demons_2d(
    registration_algorithm=registration_algorithm,
    fixed_image=fixed_gray,
    moving_image=moving_gray,
    shrink_factors=shrink_factors,
    smoothing_sigmas=smoothing_sigmas,
)


# resample using the transform we get
registered_image = sitk.Resample(moving_gray, fixed_gray, tx, sitk.sitkNearestNeighbor, 40.0)

# plotting
fixed_arr = sitk.GetArrayFromImage(fixed_gray).astype(np.float64) / 255.0
moving_arr = sitk.GetArrayFromImage(moving_gray).astype(np.float64) / 255.0

registered_arr = sitk.GetArrayFromImage(registered_image)




images = [fixed_arr, moving_arr, registered_arr]
titles = ["Fixed Image", "Moving Image", "Registered Image"]


fig, axes = plt.subplots(1, 3, figsize=(18, 6))


for i in range(3):
    axes[i].imshow(images[i], cmap='gray', interpolation="none")
    axes[i].set_title(titles[i])
    axes[i].axis('off')


fig.suptitle("SITK", fontsize=16, y=0.95)

plt.tight_layout()
plt.show()



