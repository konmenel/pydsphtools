Module pydsphtools.io
=====================
A module which handles input and output operation for DualSPHysics files.

Classes
-------

`Array(name: str, hide: bool, array_type: DataType, count: int, array_size: int, data: np.ndarray)`
:   

    ### Static methods

    `from_bytes(bytes: bytes, endianness: Endianness) ‑> pydsphtools._io.Array`
    :

    `from_stream(byte_stream: io.BytesIO, endianness: Endianness) ‑> pydsphtools._io.Array`
    :

    ### Instance variables

    `array_size: int`
    :

    `array_type: pydsphtools._io.DataType`
    :

    `count: int`
    :

    `data: numpy.ndarray`
    :

    `hide: bool`
    :

    `name: str`
    :

`Bi4File(filepath: str | os.PathLike, load_arrays: bool = False)`
:   

    ### Class variables

    `filepath: str | os.PathLike`
    :

    ### Instance variables

    `filepeth: str | os.PathLike`
    :

    `main_item: pydsphtools._io.Item`
    :

    `title: str`
    :

    ### Methods

    `get_value_by_name(self, name: str) ‑> pydsphtools._io.Value | None`
    :

`DataType(*args, **kwds)`
:   Create a collection of name/value pairs.
    
    Example enumeration:
    
    >>> class Color(Enum):
    ...     RED = 1
    ...     BLUE = 2
    ...     GREEN = 3
    
    Access them by:
    
    - attribute access:
    
      >>> Color.RED
      <Color.RED: 1>
    
    - value lookup:
    
      >>> Color(1)
      <Color.RED: 1>
    
    - name lookup:
    
      >>> Color['RED']
      <Color.RED: 1>
    
    Enumerations can be iterated over, and know how many members they have:
    
    >>> len(Color)
    3
    
    >>> list(Color)
    [<Color.RED: 1>, <Color.BLUE: 2>, <Color.GREEN: 3>]
    
    Methods can be added to enumerations, and members can have their own
    attributes -- see the documentation for details.

    ### Ancestors (in MRO)

    * enum.Enum

    ### Class variables

    `bool`
    :

    `char`
    :

    `double`
    :

    `double3`
    :

    `float`
    :

    `float3`
    :

    `int`
    :

    `int3`
    :

    `llong`
    :

    `null`
    :

    `short`
    :

    `text`
    :

    `uchar`
    :

    `uint`
    :

    `uint3`
    :

    `ullong`
    :

    `ushort`
    :

    ### Static methods

    `from_bytes(bytes: bytes, endianness: Endianness) ‑> pydsphtools._io.DataType`
    :

    ### Methods

    `is_scalar(self) ‑> bool`
    :

    `is_vector(self) ‑> bool`
    :

    `to_python_type(self) ‑> type`
    :

`Endianness(*args, **kwds)`
:   Create a collection of name/value pairs.
    
    Example enumeration:
    
    >>> class Color(Enum):
    ...     RED = 1
    ...     BLUE = 2
    ...     GREEN = 3
    
    Access them by:
    
    - attribute access:
    
      >>> Color.RED
      <Color.RED: 1>
    
    - value lookup:
    
      >>> Color(1)
      <Color.RED: 1>
    
    - name lookup:
    
      >>> Color['RED']
      <Color.RED: 1>
    
    Enumerations can be iterated over, and know how many members they have:
    
    >>> len(Color)
    3
    
    >>> list(Color)
    [<Color.RED: 1>, <Color.BLUE: 2>, <Color.GREEN: 3>]
    
    Methods can be added to enumerations, and members can have their own
    attributes -- see the documentation for details.

    ### Ancestors (in MRO)

    * enum.Enum

    ### Class variables

    `big`
    :

    `little`
    :

    ### Static methods

    `from_bytes(bytes: bytes) ‑> pydsphtools._io.Endianness`
    :

`Item(item_size: int, name: str, hide: bool, hide_values: bool, fmt_float: str, fmt_double: str, num_arrays: int, num_items: int, size_values: int, values: list[Value], items: list[Item], arrays: list[Array])`
:   

    ### Static methods

    `from_bytes(bytes: bytes, endianness: Endianness) ‑> pydsphtools._io.Item`
    :

    `from_stream(bytes_stream: io.BytesIO, endianness: Endianness) ‑> pydsphtools._io.Item`
    :

    ### Instance variables

    `arrays: list[pydsphtools._io.Array]`
    :

    `fmt_double: str`
    :

    `fmt_float: str`
    :

    `hide: bool`
    :

    `hide_values: bool`
    :

    `item_size: int`
    :

    `items: list[pydsphtools._io.Item]`
    :

    `name: str`
    :

    `num_arrays: int`
    :

    `num_items: int`
    :

    `size_values: int`
    :

    `values: list[pydsphtools._io.Value]`
    :

    ### Methods

    `get_value_by_name(self, name: str) ‑> pydsphtools._io.Value | None`
    :

    `pretty_print(self, indent=0, indent_str='  ') ‑> str`
    :

`Value(name: str, value_type: DataType, value: None | bool | str | int | float | tuple[float, float, float] | tuple[int, int, int])`
:   

    ### Static methods

    `from_bytes(bytes: bytes, endianness: Endianness) ‑> pydsphtools._io.Value`
    :

    `from_stream(stream: io.BytesIO, endianness: Endianness) ‑> tuple[str, None | bool | str | int | float | tuple[float, float, float]]`
    :

    ### Instance variables

    `name: str`
    :

    `value: None | bool | str | int | float | tuple[float, float, float] | tuple[int, int, int]`
    :

    `value_type: pydsphtools._io.DataType`
    :

    ### Methods

    `pretty_print(self, indent=0, indent_str='  ') ‑> str`
    :