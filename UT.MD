# Unit Test Status

The unit tests defined in `cardiac_mri_pipeline/registration/test_image_registration_module.py` have been written to cover the core functionalities of the `ImageRegistrationModule`, including:

- SIFT feature detection
- Mutual Information based initial rigid alignment
- B-Spline FFD based deformation field estimation
- B-Spline interpolation for applying deformation fields
- A full registration pipeline combining these advanced techniques

**However, these unit tests could not be executed or validated.**

Persistent PyTorch environment errors (`ModuleNotFoundError: No module named 'torch._strobelight'`) and underlying disk space limitations (preventing successful reinstallation of PyTorch and its components) have made it impossible to run the tests or the `ImageRegistrationModule` itself.

The tests are structured using Python's `unittest` framework and include basic assertions for each tested component. They are intended to be run once the environment issues are resolved.
