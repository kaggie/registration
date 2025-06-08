# TODO List

## Registration
- [x] Implement advanced image registration techniques (SIFT for feature detection, Mutual Information for initial rigid alignment, B-Spline FFD for deformation field estimation, and B-Spline interpolation for transformation application).
  - Details of the implementation can be found in `readme.md` and the `ImageRegistrationModule` class in `cardiac_mri_pipeline/registration/image_registration_module.py`.
- **Note**: Full runtime validation of the registration pipeline and its unit tests (`cardiac_mri_pipeline/registration/test_image_registration_module.py`) was prevented by persistent PyTorch environment errors (`ModuleNotFoundError: No module named 'torch._strobelight'`) and disk space limitations. Please refer to `UT.MD` for more information on the unit test validation status.

## Other Modules
(Placeholder for other future tasks, if any)
