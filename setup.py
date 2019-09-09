import setuptools

with open("readme.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="garf",
    version="0.0.2",
    author="David Sarrut",
    author_email="david.sarrut@creatis.insa-lyon.fr",
    description="Utility tools for GATE ARF simulations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dsarrut/garf",
    packages=['garf'],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ),
    scripts=[
        'bin/garf_compare_image_profile',
        'bin/garf_scale_and_Poisson_noise',
        'bin/garf_plot_training_dataset',
        'bin/garf_plot_test_dataset',
        'bin/garf_convert_pth_to_pt',
        'bin/garf_train',
        'bin/garf_nn_info',
        'bin/garf_build_arf_image_with_nn',
        'bin/garf_merge_images',
        'bin/garf_uncertainty_arf',
        'bin/pytorch_info',
        'bin/garf_uncertainty_analog']
)
