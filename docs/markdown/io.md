Module pydsphtools.io
=====================
A module which handles input and output operation for DualSPHysics files.

Classes
-------

`Array(name: str, hide: bool, array_type: DataType, count: int, array_size: int, data: np.ndarray)`
:   Class that represents the a bi4 array.
    
    Attributes
    ----------
    name : str
        Name of the array.
    hide : bool
        Wheither or not the array is hidden.
    array_type : DataType
        The type of the array data.
    count : int
        The number of elements in the array.
    array_size : int
        The size of the array in bytes.
    data : np.ndarray
        The data of the array.

    ### Static methods

    `from_bytes(bytes: bytes, endianness: Endianness) ‑> pydsphtools._io.Array`
    :   Constructor from bytes.
        
        Parameters
        ----------
        bytes : bytes
            The byte array.
        endianness : Endianness
            The endianness of the bytes.
        
        Returns
        -------
        Array
            The new object.

    `from_stream(byte_stream: io.BytesIO, endianness: Endianness) ‑> pydsphtools._io.Array`
    :   Constructor from bytes.
        
        Parameters
        ----------
        bytes : bytes
            The byte stream.
        endianness : Endianness
            The endianness of the bytes.
        
        Returns
        -------
        Array
            The new object.

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

    ### Methods

    `pretty_print(self, indent=0, indent_str='  ') ‑> str`
    :

`Bi4File(filepath: str | os.PathLike, load_arrays: bool = False)`
:   

    ### Ancestors (in MRO)

    * pydsphtools._io.Item

    ### Instance variables

    `filepath: str | os.PathLike`
    :

    `main_item: Item`
    :

    `title: str`
    :

`DataType(*values)`
:   Enum that handles the data type

    ### Ancestors (in MRO)

    * enum.Enum

    ### Class variables

    `bool`
    :   The type of the None singleton.

    `char`
    :   The type of the None singleton.

    `double`
    :   The type of the None singleton.

    `double3`
    :   The type of the None singleton.

    `float`
    :   The type of the None singleton.

    `float3`
    :   The type of the None singleton.

    `int`
    :   The type of the None singleton.

    `int3`
    :   The type of the None singleton.

    `llong`
    :   The type of the None singleton.

    `null`
    :   The type of the None singleton.

    `short`
    :   The type of the None singleton.

    `text`
    :   The type of the None singleton.

    `uchar`
    :   The type of the None singleton.

    `uint`
    :   The type of the None singleton.

    `uint3`
    :   The type of the None singleton.

    `ullong`
    :   The type of the None singleton.

    `ushort`
    :   The type of the None singleton.

    ### Static methods

    `from_bytes(bytes: bytes, endianness: Endianness) ‑> pydsphtools._io.DataType`
    :   Constructor from bytes.
        
        Parameters
        ----------
        bytes : bytes
            The byte array.
        endianness : Endianness
            The endianness of the bytes.
        
        Returns
        -------
        DataType
            The new object.

    ### Methods

    `is_scalar(self) ‑> bool`
    :

    `is_vector(self) ‑> bool`
    :

    `to_python_type(self) ‑> type`
    :   Converts the enum value to a python type
        
        Returns
        -------
        type
            The follow mapping is used:\
            - "null" => `None`\
            - "text", "char", "uchar" => `str`\
            - "bool" => `bool`\
            - "short", "ushort", "int", "uint", "long" "ulong" => `int`\
            - "float", "double" => `float`\
            - "int3", "uint3", "float3", "double3" => tuple\

`Endianness(*values)`
:   Enum that represents the endianness.

    ### Ancestors (in MRO)

    * enum.Enum

    ### Class variables

    `big`
    :   The type of the None singleton.

    `little`
    :   The type of the None singleton.

    ### Static methods

    `from_bytes(bytes: bytes) ‑> pydsphtools._io.Endianness`
    :   Constructor from bytes arrays
        
        Parameters
        ----------
        bytes : bytes
            The byte array
        
        Returns
        -------
        Endianness
            The new object

`Item(item_size: int, name: str, hide: bool, hide_values: bool, fmt_float: str, fmt_double: str, num_arrays: int, num_items: int, size_values: int, values: list[Value], items: list[Item], arrays: list[Array])`
:   

    ### Descendants

    * pydsphtools._io.Bi4File

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

    `get_array_by_name(self, name: str) ‑> Optional[pydsphtools._io.Array]`
    :

    `get_item_by_name(self, name: str) ‑> Optional[pydsphtools._io.Item]`
    :

    `get_value_by_name(self, name: str) ‑> Optional[pydsphtools._io.Value]`
    :

    `pretty_print(self, indent=0, indent_str='  ') ‑> str`
    :

`Value(name: str, value_type: DataType, value: None | bool | str | int | float | tuple[float, float, float] | tuple[int, int, int])`
:   

    ### Static methods

    `from_bytes(bytes: bytes, endianness: Endianness) ‑> pydsphtools._io.Value`
    :

    `from_stream(stream: io.BytesIO, endianness: Endianness) ‑> pydsphtools._io.Value`
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