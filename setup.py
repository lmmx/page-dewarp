from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    reqs = fh.read().splitlines()


def local_scheme(version):
    return ""


def version_scheme(version):
    return version.tag.base_version


setup(
    name="page_dewarp",
    author="Louis Maddox",
    author_email="louismmx@gmail.com",
    description=("Page dewarping and thresholding using a cubic sheet model."),
    license="MIT License",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lmmx/page-dewarp",
    packages=find_packages("src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
        "Topic :: Multimedia :: Graphics :: Graphics Conversion",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    include_package_data=True,
    use_scm_version={
        "write_to": "version.py",
        "version_scheme": version_scheme,
        "local_scheme": local_scheme,
    },
    setup_requires=["setuptools_scm"],
    entry_points={
        "console_scripts": [
            "page-dewarp = page_dewarp.__main__:main",
        ],
    },
    install_requires=reqs,
    python_requires=">=3",
)
