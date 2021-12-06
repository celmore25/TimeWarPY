from setuptools import setup, find_packages


requirements = ['pandas>=1.0.0']

setup(
    name="timewarpy",
    version="0.0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
)
