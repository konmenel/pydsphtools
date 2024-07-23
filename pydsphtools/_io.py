from __future__ import annotations
import os
import io
import json
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
#   - "VALUES"
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
# - uint32 str_size (5)
# - "ARRAY"
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

    def __init__(self) -> None:
        raise NotImplementedError

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
    values: dict[str, None | bool | str | int | float | tuple[float, float, float]]
    items: list[Item]
    # Arrays should not be loaded as they can be too big
    # arrays: list[Array]

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
        values: dict[str, None | bool | str | int | float | tuple[float, float, float]],
        items: list[Item],
        # arrays: list[Array],
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
        # self._arrays = arrays

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
        values = {
            name: value
            for _ in range(num_values)
            for name, value in [Item._parse_value_from_stream(stream, endianness)]
        }
        assert stream.read() == b"", "Values buffer is not empty."

        # Update buffer since we reached the end of values
        buff = bytes_stream.read(size_values)
        stream = io.BytesIO(buff)
        items = [Item.from_stream(stream, endianness) for _ in range(num_items)]

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
        )

    @classmethod
    def from_bytes(cls, bytes: bytes, endianness: Endianness) -> Item:
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
    def values(
        self,
    ) -> dict[str, None | bool | str | int | float | tuple[float, float, float]]:
        return self._values

    @property
    def items(self) -> list[Item]:
        return self._items

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
            f"{indent_str*(indent+1)}values = {{\n"
            f"{self._pretty_print_dict(self.values, indent=indent+1)}"
            f"{indent_str*(indent+1)}}}\n"
        )
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

    @staticmethod
    def _parse_value_from_stream(
        stream: io.BytesIO,
        endianness: Endianness,
    ) -> tuple[str, None | bool | str | int | float | tuple[float, float, float]]:
        name_size = int.from_bytes(stream.read(UINT_SIZE), endianness.name)
        name = stream.read(name_size).decode("utf-8")
        data_type = DataType(int.from_bytes(stream.read(INT_SIZE), endianness.name))
        edn_fmt = "<" if endianness == Endianness.little else ">"

        match data_type:
            case DataType.null:
                return name, None
            case DataType.char | DataType.uchar:
                return name, stream.read(CHAR_SIZE).decode("utf-8")
            case DataType.text:
                str_size = int.from_bytes(stream.read(UINT_SIZE), endianness.name)
                return name, stream.read(str_size).decode("utf-8")
            case DataType.short | DataType.ushort:
                return name, int.from_bytes(stream.read(SHORT_SIZE), endianness.name)
            case DataType.int | DataType.uint | DataType.bool:
                return name, int.from_bytes(stream.read(INT_SIZE), endianness.name)
            case DataType.llong | DataType.ullong:
                return name, int.from_bytes(stream.read(LONG_SIZE), endianness.name)
            case DataType.float:
                return name, struct.unpack(f"{edn_fmt}f", stream.read(FLOAT_SIZE))[0]
            case DataType.double:
                return name, struct.unpack(f"{edn_fmt}d", stream.read(DOUBLE_SIZE))[0]
            case DataType.int3 | DataType.uint3:
                return name, (
                    int.from_bytes(stream.read(INT_SIZE), endianness.name),
                    int.from_bytes(stream.read(INT_SIZE), endianness.name),
                    int.from_bytes(stream.read(INT_SIZE), endianness.name),
                )
            case DataType.float3:
                return name, struct.unpack(f"{edn_fmt}3f", stream.read(FLOAT3_SIZE))
            case DataType.double3:
                return name, struct.unpack(f"{edn_fmt}3d", stream.read(DOUBLE3_SIZE))
            case _:
                raise NotImplementedError(
                    f"Can't parse value of data type {data_type}"
                    " because it is not implemented"
                )


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
