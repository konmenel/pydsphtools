Module pydsphtools.exceptions
=============================
The module with the definition of the exceptions used in the packages.

Classes
-------

`DSPHBinaryNotFound(binary_name: str, path: str, *args: object)`
:   Raised if a binary is not found in the path.
    
    Attributes
    ----------
    binary_name : str
        Name of the binary
    path : str
        The path that was searched.

    ### Ancestors (in MRO)

    * builtins.Exception
    * builtins.BaseException

    ### Class variables

    `binary_name: str`
    :

    `path: str`
    :

`InvalidTimeInterval(tmin: float, tmax: float)`
:   Raised when a variable is not found in the output file.
    
    Attributes
    ----------
    tmin : float
        The lower bound of the time interval.
    tmax : float
        The higher bound of the time interval.

    ### Ancestors (in MRO)

    * builtins.Exception
    * builtins.BaseException

    ### Class variables

    `tmax: float`
    :

    `tmin: float`
    :

`MissingEnvironmentVariable(var_name: str, *args)`
:   Raised if an enviroment variable is undefined.
    
    Attributes
    ----------
        var_name : str
            The name of the environment variable.

    ### Ancestors (in MRO)

    * builtins.Exception
    * builtins.BaseException

    ### Class variables

    `var_name: str`
    :

`NotFoundInOutput(missing: str, filename: str = 'Run.out')`
:   Raised when a variable is not found in the output file.
    
    Attributes
    ----------
    missing : str
        What was not found.
    filename : str, optional
        The name of the output file (either `Run.out` or `Run.csv`). By default,
         `Run.out`.

    ### Ancestors (in MRO)

    * builtins.Exception
    * builtins.BaseException

    ### Class variables

    `filename: str`
    :

    `missing: str`
    :

`UnsupportedPlatform(platform: str, *args: object)`
:   Raised if the platform is not supperted by DualSPHysics/
    
    Attributes
    ----------
    platform : str
        Name of the platform

    ### Ancestors (in MRO)

    * builtins.Exception
    * builtins.BaseException

    ### Class variables

    `platform: str`
    :