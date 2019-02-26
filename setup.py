import setuptools

with open('README.md', 'r') as file:
    long_description = file.read()

setuptools.setup(
    name='mlots',
    version='0.0.0.a2',
    author="Vivek Mahato",
    author_email="vivek.mahato@ucdconnect.ie",
    description="Machine Learning Over Time-Series: A toolkit for time-series analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/vivekmahato/mlots',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
