from __future__ import annotations
import os
import io
import struct
from enum import Enum


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

    @staticmethod
    def from_bytes(bytes: bytes) -> Endianness:
        return Endianness(int.from_bytes(bytes))


class Array:
    name: str
    hide: bool
    array_type: DataType
    count: int
    array_size: int
    data: list[None | bool | str | int | float | tuple[float, float, float]]


class Item:
    name: str
    hide: bool
    hide_values: bool
    fmt_float: str
    fmt_double: str
    num_arrays: int
    num_items: int
    size_values: int
    values: list[None | bool | str | int | float | tuple[float, float, float]]
    items: list[Item]
    arrays: list[Array]

    def __init__(
        self,
        name: str,
        hide: bool,
        hide_values: bool,
        fmt_float: str,
        fmt_double: str,
        num_arrays: int,
        num_items: int,
        size_values: int,
        values: dict[str, None | bool | str | int | float | tuple[float, float, float]],
        # items: list[Item],
        # arrays: list[Array],
    ) -> None:
        self._name = name
        self._hide = hide
        self._hide_values = hide_values
        self._fmt_float = fmt_float
        self._fmt_double = fmt_double
        self._num_arrays = num_arrays
        self._num_items = num_items
        self._size_values = size_values
        self._values = values
        # self._items = items
        # self._arrays = arrays

    @classmethod
    def from_stream(cls: Item, stream: io.BytesIO, endianness: Endianness) -> Item:
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

        values_str_size = int.from_bytes(stream.read(UINT_SIZE), endianness.name)
        assert values_str_size == 7
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
        return cls(
            name,
            hide,
            hide_values,
            fmt_float,
            fmt_double,
            num_arrays,
            num_items,
            size_values,
            values,
        )

    @property
    def name(self) -> str:
        return self._name

    def __repr__(self) -> str:
        return (
            f"Item(name={self._name},hide={self._hide},hide_values={self._hide_values},"
            f"fmt_float={self._fmt_float},fmt_double={self._fmt_double},"
            f"num_arrays={self._num_arrays},num_items={self._num_items},"
            f"size_values={self._size_values},values={self._values})"
        )

    @staticmethod
    def _parse_value_from_stream(
        stream: io.BytesIO,
        endianness: Endianness,
    ) -> tuple[str, None | bool | str | int | float | tuple[float, float, float]]:
        name_size = int.from_bytes(stream.read(UINT_SIZE), endianness.name)
        name = stream.read(name_size).decode("utf-8")
        data_type = DataType(int.from_bytes(stream.read(INT_SIZE), endianness.name))

        match data_type:
            case DataType.null:
                return name, None
            case DataType.bool:
                return name, bool.from_bytes(stream.read(UINT_SIZE), endianness.name)
            case DataType.char | DataType.uchar:
                return name, stream.read(CHAR_SIZE).decode("utf-8")
            case DataType.text:
                str_size = int.from_bytes(stream.read(UINT_SIZE), endianness.name)
                return name, stream.read(str_size).decode("utf-8")
            case DataType.short | DataType.ushort:
                return name, int.from_bytes(stream.read(SHORT_SIZE), endianness.name)
            case DataType.int | DataType.uint:
                return name, int.from_bytes(stream.read(INT_SIZE), endianness.name)
            case DataType.llong | DataType.ullong:
                return name, int.from_bytes(stream.read(LONG_SIZE), endianness.name)
            case DataType.float:
                return name, struct.unpack("f", stream.read(FLOAT_SIZE))
            case DataType.double:
                return name, struct.unpack("d", stream.read(DOUBLE_SIZE))
            case DataType.int3 | DataType.uint3:
                return name, (
                    int.from_bytes(stream.read(INT_SIZE), endianness.name),
                    int.from_bytes(stream.read(INT_SIZE), endianness.name),
                    int.from_bytes(stream.read(INT_SIZE), endianness.name),
                )
            case DataType.float3:
                return name, (
                    struct.unpack("f", stream.read(FLOAT_SIZE)),
                    struct.unpack("f", stream.read(FLOAT_SIZE)),
                    struct.unpack("f", stream.read(FLOAT_SIZE)),
                )
            case DataType.double3:
                return name, (
                    struct.unpack("d", stream.read(DOUBLE_SIZE)),
                    struct.unpack("d", stream.read(DOUBLE_SIZE)),
                    struct.unpack("d", stream.read(DOUBLE_SIZE)),
                )
            case _:
                raise NotImplementedError(
                    f"Can't parse value of data type {data_type}"
                    " because it is not implemented"
                )


def read_bi4(partfile: str | os.PathLike) -> None:
    with open(partfile, "rb") as file:
        title = file.read(60)
        assert title[-1] == 0x00
        title = title[:-1].decode("utf-8")
        byteorder = Endianness.from_bytes(file.read(1))
        extra = file.read(3)
        print(f"{title}", end="")
        print(f"{extra}")
        size = int.from_bytes(file.read(4), byteorder.name)
        print(size)
        print(Item.from_stream(file, byteorder))


if __name__ == "__main__":
    read_bi4("Part_0000.bi4")
