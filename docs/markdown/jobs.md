Module pydsphtools.jobs
=======================
Module that handles DualSPHysics jobs. Jobs are sequential runs of the DualSPHysics solver, pre-proccessing
tools or post-proccessing tools.

Classes
-------

`Binary(name: str, *args: *tuple[str])`
:   

    ### Descendants

    * pydsphtools._jobs.Dualsphysics
    * pydsphtools._jobs.Gencase

    ### Class variables

    `args: list[str]`
    :

    `name: str`
    :

    `path: pathlib.Path`
    :

    ### Methods

    `add_args(self, *args: *tuple[str]) ‑> Self`
    :

    `get_command_list(self) ‑> list[str]`
    :

    `run(self, *args, **kwargs) ‑> Self`
    :

`Dualsphysics(*args: *tuple[str], gpu: bool = True, version: str = '5.2')`
:   

    ### Ancestors (in MRO)

    * pydsphtools._jobs.Binary

`Gencase(*args: *tuple[str])`
:   

    ### Ancestors (in MRO)

    * pydsphtools._jobs.Binary

`Job(*binaries: *tuple[Binary], verbose: bool = True)`
:   Class that that create a dualsphysics job.
    
    Attributes
    ----------
    binaries: list[Binary]
        The binary objects that will be run. Each binary will be run sequentially
        in the order that they have been passed.
    verbose : bool
        Prints information the job when `run()` is called. Default, `True`.
    
    Examples
    --------
    ```python
    from pydsphtools import Job
    
    job = Job()
    job.add_binaries(Dualsphysics("-h"), verbose=True)
    job.run()
    ```

    ### Class variables

    `binaries: list[pydsphtools._jobs.Binary]`
    :

    `verbose: bool`
    :

    ### Instance variables

    `env: dict[str, str]`
    :

    ### Methods

    `add_binaries(self, *binaries: *tuple[Binary]) ‑> Self`
    :

    `print_env(self)`
    :

    `run(self)`
    :