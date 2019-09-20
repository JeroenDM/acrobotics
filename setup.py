from setuptools import setup, find_packages, Extension
import numpy

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

with open("README.md", "r") as fh:
    long_description = fh.read()

geometry_module = Extension(
    "_geometry",
    language="c++",
    extra_compile_args=["-std=c++11"],
    sources=["src/acrobotics/cpp/geometry.i", "src/acrobotics/cpp/src/geometry.cpp"],
    include_dirs=["src/acrobotics/cpp/include", "/usr/include/eigen3", numpy.get_include()],
    swig_opts=["-c++", "-I acrobotics/cpp"],
)


setup(
    name="acrobotics",
    version="0.0.3",
    author="Jeroen De Maeyer",
    author_email="jeroen.demaeyer@kuleuven.be",
    description="Primitive robot kinematics and collision checking.",
    url="https://github.com/JeroenDM/acrobotics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    install_requires=requirements,
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    python_requires=">=3.6",
    ext_package="acrobotics.cpp",
    ext_modules=[geometry_module],
)
