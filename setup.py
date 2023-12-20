import setuptools

with open("readme.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="garf",
    version="2.5",
    author="David Sarrut",
    author_email="david.sarrut@creatis.insa-lyon.fr",
    description="Utility tools for GATE ARF simulations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dsarrut/garf",
    packages=["garf"],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ),
    install_requires=["itk", "numpy", "uproot", "pre-commit", "torch>=2.1.0", "tqdm"],
    scripts=[
        "bin/garf_compare_image_profile",
        "bin/garf_plot_training_dataset",
        "bin/garf_plot_test_dataset",
        "bin/garf_train",
        "bin/garf_nn_info",
        "bin/pytorch_info",
    ],
)
