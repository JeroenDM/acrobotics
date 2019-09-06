from setuptools import setup, find_packages

setup(
    name="acrobotics",
    version="0.0.1",
    author="Jeroen De Maeyer",
    author_email="jeroen.demaeyer@kuleuven.be",
    url="https://github.com/JeroenDM/acrobotics",
    packages=find_packages(where="src"),
    install_requires=["numpy", "matplotlib", "python-fcl", "tqdm"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
)
