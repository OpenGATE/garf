import setuptools

with open("readme.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="garf",
    version="0.0.1",
    author="David Sarrut",
    author_email="david.sarrut@creatis.insa-lyon.fr",
    description="Python tools for GATE ARF simulations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dsarrut/garf",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ),
    scripts=['bin/garf_plot_training_dataset']
)
