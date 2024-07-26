from __future__ import annotations
import os
import io
import struct
from enum import Enum
import numpy as np


# Structure of file:
# ===================================
# uint size_item_def
# - uint32 str_size (6)
# - str "\nITEM\n"
# - uint32 str_size
# - str name
# - bool hide (as int32)
# - bool hidevalues (as int32)
# - uint32 str_size
# - str fmtfloat
# - uint32 str_size
# - str fmtdouble
# - uint32 num_arrays
# - uint32 num_items
# - uint32 size_values
#   - uint32 str_size (7)
#   - "\nVALUES"
#   - uint32 num_values
#   - [value_0]
#       - uint32 str_size
#       - str name
#       - uint32 value_type (enum `DataType` value)
#       - value_type value
#   - [value_1]
#       ...
#   - [value_n]
# [item_0]
# ...
# [item_n]
# [array_0]
# uint32 size_array_def [n]
# - uint32 str_size (6)
# - "\nARRAY"
# - uint32 str_size
# - str name
# - bool hide (as int32)
# - int32 type
# - uint32 count
# - uint32 array_size
#   - [array data]
# [array_1]
# ...
# [array_n]


BOOL_SIZE: int = 1
CHAR_SIZE: int = 1
SHORT_SIZE: int = 2
USHORT_SIZE: int = 2
INT_SIZE: int = 4
UINT_SIZE: int = 4
LONG_SIZE: int = 8
ULONG_SIZE: int = 8
FLOAT_SIZE: int = 4
DOUBLE_SIZE: int = 8
INT3_SIZE: int = INT_SIZE * 3
UINT3_SIZE: int = UINT_SIZE * 3
FLOAT3_SIZE: int = FLOAT_SIZE * 3
DOUBLE3_SIZE: int = DOUBLE_SIZE * 3


class DataType(Enum):
    # bytes
    null = 0
    text = 1
    bool = 2
    char = 3
    uchar = 4

    # ints
    short = 5
    ushort = 6
    int = 7
    uint = 8
    llong = 9
    ullong = 10

    # floats
    float = 11
    double = 12

    # vectors
    int3 = 20
    uint3 = 21
    float3 = 22
    double3 = 23

    @classmethod
    def from_bytes(cls: DataType, bytes: bytes, endianness: Endianness) -> DataType:
        return cls(int.from_bytes(bytes, endianness.name))

    def to_python_type(self) -> type:
        if self == DataType.null:
            return type(None)
        elif self == DataType.bool:
            return bool
        elif self.value in (1, 3, 4):
            return str
        elif self.value in range(5, 11):
            return int
        elif self.value in (11, 12):
            return float
        elif self.value in range(20, 24):
            return tuple

    def is_scalar(self) -> bool:
        return self.value in range(5, 13)

    def is_vector(self) -> bool:
        return self.value in range(20, 24)


class Endianness(Enum):
    little = 0
    big = 1

    @classmethod
    def from_bytes(cls, bytes: bytes) -> Endianness:
        return cls(int.from_bytes(bytes))


class Array:
    name: str
    hide: bool
    array_type: DataType
    count: int
    array_size: int
    data: np.ndarray

    __slots__ = (
        "_name",
        "_hide",
        "_array_type",
        "_count",
        "_array_size",
        "_data",
    )

    def __init__(
        self,
        name: str,
        hide: bool,
        array_type: DataType,
        count: int,
        array_size: int,
        data: np.ndarray,
    ) -> None:
        self._name = name
        self._hide = hide
        self._array_type = array_type
        self._count = count
        self._array_size = array_size
        self._data = data

    @classmethod
    def from_stream(
        cls: Array, byte_stream: io.BytesIO, endianness: Endianness
    ) -> Array:
        array_def_size: int = int.from_bytes(
            byte_stream.read(UINT_SIZE), endianness.name
        )
        buf = byte_stream.read(array_def_size)
        stream = io.BytesIO(buf)

        array_str_size: int = int.from_bytes(stream.read(UINT_SIZE), endianness.name)
        assert array_str_size == 6, f"Expected 6 but found {array_str_size}"
        array_str: str = stream.read(array_str_size).decode("utf-8")
        assert array_str == "\nARRAY", f"Expected '\nARRAY' but found'{array_str}'"
        name_size: int = int.from_bytes(stream.read(UINT_SIZE), endianness.name)
        name: str = stream.read(name_size).decode("utf-8")
        hide: bool = bool.from_bytes(stream.read(UINT_SIZE), endianness.name)
        array_type: DataType = DataType.from_bytes(stream.read(INT_SIZE), endianness)
        count: int = int.from_bytes(stream.read(UINT_SIZE), endianness.name)
        array_size: int = int.from_bytes(stream.read(UINT_SIZE), endianness.name)
        assert stream.read() == b"", "Array definition buffer is not empty."
        stream.close()

        typefmt = cls._get_numpy_fmt(array_type, endianness)
        data = np.frombuffer(
            byte_stream.read(array_size),
            dtype=np.dtype(typefmt),
            count=count * 3 if array_type.is_vector() else count,
        )
        if array_type.is_vector():
            data = data.reshape((-1, 3))

        return cls(name, hide, array_type, count, array_size, data)

    @classmethod
    def from_bytes(cls: Array, bytes: bytes, endianness: Endianness) -> Array:
        return cls.from_stream(io.BytesIO(bytes), endianness)

    @property
    def name(self) -> str:
        return self._name

    @property
    def hide(self) -> bool:
        return self._hide

    @property
    def array_type(self) -> DataType:
        return self._array_type

    @property
    def count(self) -> int:
        return self._count

    @property
    def array_size(self) -> int:
        return self._array_size

    @property
    def data(self) -> np.ndarray:
        return self._data

    @staticmethod
    def _get_numpy_fmt(data_type: DataType, endianness: Endianness) -> str:
        bo = "<" if endianness == Endianness.little else ">"
        match data_type:
            case DataType.int | DataType.int3:
                return f"{bo}i{INT_SIZE}"
            case DataType.uint | DataType.uint3:
                return f"{bo}u{UINT_SIZE}"
            case DataType.short:
                return f"{bo}i{SHORT_SIZE}"
            case DataType.ushort:
                return f"{bo}u{SHORT_SIZE}"
            case DataType.llong:
                return f"{bo}i{LONG_SIZE}"
            case DataType.ullong:
                return f"{bo}i{ULONG_SIZE}"
            case DataType.float | DataType.float3:
                return f"{bo}f{FLOAT_SIZE}"
            case DataType.double | DataType.double3:
                return f"{bo}f{DOUBLE_SIZE}"
            case _:
                raise NotImplementedError(
                    f"Can't parse array of data type {data_type}"
                    " because it is not implemented."
                )


class Value:
    name: str
    value_type: DataType
    value: (
        None
        | bool
        | str
        | int
        | float
        | tuple[float, float, float]
        | tuple[int, int, int]
    )

    __slots__ = ("_name", "_value_type", "_value")

    def __init__(
        self,
        name: str,
        value_type: DataType,
        value: (
            None
            | bool
            | str
            | int
            | float
            | tuple[float, float, float]
            | tuple[int, int, int]
        ),
    ) -> None:
        self._name = name
        self._value_type = value_type
        self._value = value

    @property
    def name(self) -> str:
        return self._name

    @property
    def value_type(self) -> DataType:
        return self._value_type

    @property
    def value(
        self,
    ) -> (
        None
        | bool
        | str
        | int
        | float
        | tuple[float, float, float]
        | tuple[int, int, int]
    ):
        return self._value

    @classmethod
    def from_stream(
        cls: Value,
        stream: io.BytesIO,
        endianness: Endianness,
    ) -> tuple[str, None | bool | str | int | float | tuple[float, float, float]]:
        name_size = int.from_bytes(stream.read(UINT_SIZE), endianness.name)
        name = stream.read(name_size).decode("utf-8")
        data_type = DataType(int.from_bytes(stream.read(INT_SIZE), endianness.name))
        edn_fmt = "<" if endianness == Endianness.little else ">"

        match data_type:
            case DataType.null:
                data = None
            case DataType.char | DataType.uchar:
                data = stream.read(CHAR_SIZE).decode("utf-8")
            case DataType.text:
                str_size = int.from_bytes(stream.read(UINT_SIZE), endianness.name)
                data = stream.read(str_size).decode("utf-8")
            case DataType.short | DataType.ushort:
                data = int.from_bytes(stream.read(SHORT_SIZE), endianness.name)
            case DataType.int | DataType.uint | DataType.bool:
                data = int.from_bytes(stream.read(INT_SIZE), endianness.name)
            case DataType.llong | DataType.ullong:
                data = int.from_bytes(stream.read(LONG_SIZE), endianness.name)
            case DataType.float:
                data = struct.unpack(f"{edn_fmt}f", stream.read(FLOAT_SIZE))[0]
            case DataType.double:
                data = struct.unpack(f"{edn_fmt}d", stream.read(DOUBLE_SIZE))[0]
            case DataType.int3 | DataType.uint3:
                data = (
                    int.from_bytes(stream.read(INT_SIZE), endianness.name),
                    int.from_bytes(stream.read(INT_SIZE), endianness.name),
                    int.from_bytes(stream.read(INT_SIZE), endianness.name),
                )
            case DataType.float3:
                data = struct.unpack(f"{edn_fmt}3f", stream.read(FLOAT3_SIZE))
            case DataType.double3:
                data = struct.unpack(f"{edn_fmt}3d", stream.read(DOUBLE3_SIZE))
            case _:
                raise NotImplementedError(
                    f"Can't parse value of data type {data_type}"
                    " because it is not implemented"
                )

        return cls(name, data_type, data)

    @classmethod
    def from_bytes(cls: Value, bytes: bytes, endianness: Endianness) -> Value:
        return cls.from_stream(io.BytesIO(bytes), endianness)

    def pretty_print(self, indent=0, indent_str="  ") -> str:
        return f"{indent_str * indent}{self}"

    def __str__(self) -> str:
        return f"{self.name}: {self. value}"


class Item:
    item_size: int
    name: str
    hide: bool
    hide_values: bool
    fmt_float: str
    fmt_double: str
    num_arrays: int
    num_items: int
    size_values: int
    values: list[Value]
    items: list[Item]
    # TODO: Arrays should not be loaded by default as they can be too big
    arrays: list[Array]

    __slots__ = (
        "_item_size",
        "_name",
        "_hide",
        "_hide_values",
        "_fmt_float",
        "_fmt_double",
        "_num_arrays",
        "_num_items",
        "_size_values",
        "_values",
        "_items",
        "_arrays",
    )

    def __init__(
        self,
        item_size: int,
        name: str,
        hide: bool,
        hide_values: bool,
        fmt_float: str,
        fmt_double: str,
        num_arrays: int,
        num_items: int,
        size_values: int,
        values: list[Value],
        items: list[Item],
        arrays: list[Array],
    ) -> None:
        self._item_size = item_size
        self._name = name
        self._hide = hide
        self._hide_values = hide_values
        self._fmt_float = fmt_float
        self._fmt_double = fmt_double
        self._num_arrays = num_arrays
        self._num_items = num_items
        self._size_values = size_values
        self._values = values
        self._items = items
        self._arrays = arrays

    @classmethod
    def from_stream(
        cls: Item, bytes_stream: io.BytesIO, endianness: Endianness
    ) -> Item:
        item_size = int.from_bytes(bytes_stream.read(4), endianness.name)
        buff = bytes_stream.read(item_size)
        stream = io.BytesIO(buff)
        item_str_size = int.from_bytes(stream.read(UINT_SIZE), endianness.name)
        assert item_str_size == 6, f"Expected 6 but found {item_str_size}"
        item_str = stream.read(item_str_size).decode("utf-8")
        assert item_str == "\nITEM\n", f"Expected '\\nITEM\\n' but found '{item_str}'"
        name_size = int.from_bytes(stream.read(UINT_SIZE), endianness.name)
        name = stream.read(name_size).decode("utf-8")
        hide = bool.from_bytes(stream.read(UINT_SIZE), endianness.name)
        hide_values = bool.from_bytes(stream.read(UINT_SIZE), endianness.name)
        fmt_float_size = int.from_bytes(stream.read(UINT_SIZE), endianness.name)
        fmt_float = stream.read(fmt_float_size).decode("utf-8")
        fmt_double_size = int.from_bytes(stream.read(UINT_SIZE), endianness.name)
        fmt_double = stream.read(fmt_double_size).decode("utf-8")
        num_arrays = int.from_bytes(stream.read(UINT_SIZE), endianness.name)
        num_items = int.from_bytes(stream.read(UINT_SIZE), endianness.name)
        size_values = int.from_bytes(stream.read(UINT_SIZE), endianness.name)
        assert stream.read() == b"", "Item buffer is not empty."
        stream.close()

        # Update buffer since we the items definition is finished
        buff = bytes_stream.read(size_values)
        stream = io.BytesIO(buff)
        values_str_size = int.from_bytes(stream.read(UINT_SIZE), endianness.name)
        assert values_str_size == 7, f"Expected 7 but found {values_str_size}"
        values_str = stream.read(values_str_size).decode("utf-8")
        assert (
            values_str == "\nVALUES"
        ), f"Expected '\\nVALUES' but found '{values_str}'"
        num_values = int.from_bytes(stream.read(UINT_SIZE), endianness.name)
        values = [Value.from_stream(stream, endianness) for _ in range(num_values)]
        assert stream.read() == b"", "Values buffer is not empty."
        stream.close()

        # Update buffer since we reached the end of values
        items = [Item.from_stream(bytes_stream, endianness) for _ in range(num_items)]
        arrays = [
            Array.from_stream(bytes_stream, endianness) for _ in range(num_arrays)
        ]

        return cls(
            item_size,
            name,
            hide,
            hide_values,
            fmt_float,
            fmt_double,
            num_arrays,
            num_items,
            size_values,
            values,
            items,
            arrays,
        )

    @classmethod
    def from_bytes(cls: Item, bytes: bytes, endianness: Endianness) -> Item:
        return cls.from_stream(io.BytesIO(bytes), endianness)

    @property
    def item_size(self) -> int:
        return self._item_size

    @property
    def name(self) -> str:
        return self._name

    @property
    def hide(self) -> bool:
        return self._hide

    @property
    def hide_values(self) -> bool:
        return self._hide_values

    @property
    def fmt_float(self) -> str:
        return self._fmt_float

    @property
    def fmt_double(self) -> str:
        return self._fmt_double

    @property
    def num_arrays(self) -> int:
        return self.num_arrays

    @property
    def num_items(self) -> int:
        return self._num_items

    @property
    def size_values(self) -> int:
        return self._size_values

    @property
    def values(self) -> list[Value]:
        return self._values

    @property
    def items(self) -> list[Item]:
        return self._items

    @property
    def arrays(self) -> list[Array]:
        return self._arrays

    def __str__(self) -> str:
        return self.pretty_print()

    def __repr__(self) -> str:
        return self.pretty_print()

    def pretty_print(self, indent=0, indent_str="  ") -> str:
        ret = (
            f"{indent_str*indent}Item(\n"
            f"{indent_str*(indent+1)}item_size = {self._item_size},\n"
            f"{indent_str*(indent+1)}name = {self._name},\n"
            f"{indent_str*(indent+1)}hide = {self._hide},\n"
            f"{indent_str*(indent+1)}hide_values = {self._hide_values},\n"
            f"{indent_str*(indent+1)}fmt_float = {self._fmt_float},\n"
            f"{indent_str*(indent+1)}fmt_double = {self._fmt_double},\n"
            f"{indent_str*(indent+1)}num_arrays = {self._num_arrays},\n"
            f"{indent_str*(indent+1)}num_items = {self._num_items},\n"
            f"{indent_str*(indent+1)}size_values = {self._size_values},\n"
            f"{indent_str*(indent+1)}values = [\n"
        )
        for value in self._values:
            ret += f"{value.pretty_print(indent=indent+2)}\n"
        ret += f"{indent_str*(indent+1)}]\n"

        if self._num_items:
            ret += f"{indent_str*(indent+1)}items = [\n"
            for item in self._items:
                ret += f"{indent_str*indent}{item.pretty_print(indent+2)}"
            ret += f"{indent_str*(indent+1)}]\n"
        else:
            ret += f"{indent_str*(indent+1)}items = []\n"
        ret += f"{indent_str*indent})\n"
        return ret

    def _pretty_print_dict(self, d: dict, indent=0, indent_str="  ") -> str:
        ret = ""
        for key, value in d.items():
            ret += f"{indent_str*(indent+1)}{key}: "
            if isinstance(value, dict):
                ret += "{\n"
                ret += f"{self._pretty_print_dict(value, indent+1)}"
                ret += f"{indent_str*(indent+1)}}}\n"
            else:
                ret += f"{value}\n"
        return ret


class Bi4File:
    filename: str
    item: Item

    def __init__(self, filepath: str | os.PathLike) -> None:
        raise NotImplementedError


def read_bi4(partfile: str | os.PathLike) -> None:
    with open(partfile, "rb") as file:
        title = file.read(60)
        assert title[-1] == 0x00
        title = title[:-1].decode("utf-8")
        byteorder = Endianness.from_bytes(file.read(1))
        extra = file.read(3)
        print(f"{title}", end="")
        print(f"{extra}")
        main_item = Item.from_stream(file, byteorder)
        print(main_item)


if __name__ == "__main__":
    read_bi4("Part_0000.bi4")
