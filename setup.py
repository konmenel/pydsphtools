import setuptools

setuptools.setup(
    name="pydsphtools",
    version="0.1",
    description="Useful functions for post processing DualSPHysics cases",
    author="Constantinos Menelaou",
    install_requires=["pandas", "numpy"],
    packages=setuptools.find_packages(),
    zip_safe=False,
)
