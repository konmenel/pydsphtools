import setuptools

setuptools.setup(
    name="pydsphtools",
    version="1.0",
    description="Useful functions for post processing DualSPHysics cases",
    author="Constantinos Menelaou",
    license='GNU',
    install_requires=["pandas", "numpy", "scipy", "lxml"],
    packages=setuptools.find_packages(),
    zip_safe=False,
    extras_require={
        "dev": ["black", "flake8", "pdoc3"],
    },
)
