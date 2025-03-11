import SimpleITK as sitk
import numpy as np


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
        sitk.sitkBSpline,  # could change to other methods
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
            fixed_images[-1].GetSize(),  # 此时 fixed_gray 是二维的 (宽, 高)
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

# examples on how to set config
demons_filter = sitk.SymmetricForcesDemonsRegistrationFilter()
demons_filter.SetNumberOfIterations(20)
demons_filter.SetSmoothDisplacementField(True)
demons_filter.SetStandardDeviations(2.0)

demons_filter.AddCommand(sitk.sitkIterationEvent, lambda: iteration_callback(demons_filter))

