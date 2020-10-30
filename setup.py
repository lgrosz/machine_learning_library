import setuptools

# todo
#with open("README.md", "r") as fh:
#    long_description = fh.read()

setuptools.setup(
    name="myml",
    version="0.0.1",
    author="Logan Grosz",
    author_email="logan.grosz@gmail.com",
    description="A small machine learning library",
    #long_description=long_description
    #long_description_content_type="text/markdown",
    url="https://github.com/logbaseaofn/machine_learning_library",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
