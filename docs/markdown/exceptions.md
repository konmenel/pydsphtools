Module pydsphtools.exceptions
=============================
The module with the definition of the exceptions used in the packages.

Classes
-------

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

`MissingEnvironmentVariable(var_name: str, *args)`
:   Common base class for all non-exit exceptions.

    ### Ancestors (in MRO)

    * builtins.Exception
    * builtins.BaseException

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