import setuptools

setuptools.setup(
    name="pydsphtools",
    version="1.0",
    description="Useful functions for post processing DualSPHysics cases",
    author="Constantinos Menelaou",
    license='GNU',
    install_requires=["pandas", "numpy"],
    packages=setuptools.find_packages(),
    zip_safe=False,
)
