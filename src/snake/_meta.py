"""Meta Class Voodoo magic."""

from __future__ import annotations
import os
import dataclasses
import itertools
import logging
import sys
from collections import defaultdict
from collections.abc import Callable, Iterator, Iterable
from enum import Enum, EnumMeta
from functools import wraps
from typing import Any, TypeVar
from functools import partial

from typing_extensions import dataclass_transform

T = TypeVar("T")

ThreeInts = tuple[int, int, int]
ThreeFloats = tuple[float, float, float] | list[float]


def make_log_property(dunder_name: str) -> Callable:
    """Create a property logger."""

    def log(self: Any) -> logging.Logger:
        """Get a logger."""
        return logging.getLogger(f"{dunder_name}.{self.__class__.__name__}")

    return log


def dataclass_repr_html(obj: Any, vertical: bool = True) -> str:
    """
    Recursive HTML representation for dataclasses.

    This function generates an HTML table representation of a dataclass,
    including nested dataclasses.

    Parameters
    ----------
    obj: The dataclass instance.

    Returns
    -------
        str: An HTML table string representing the dataclass.
    """
    class_name = obj.__class__.__name__
    table_rows = [
        '<table style="border:1px solid lightgray;">'
        '<caption style="border:1px solid lightgray;">'
        f"<strong>{class_name}</strong></caption>"
    ]
    from dataclasses import fields

    resolved_hints = obj.__annotations__
    field_names = [f.name for f in fields(obj)]
    field_values = {name: getattr(obj, name) for name in field_names}
    resolved_field_types = {name: resolved_hints[name] for name in field_names}

    if vertical:  # switch between vertical and horizontal mode
        for field_name in field_names:
            # Recursively call _repr_html_ for nested dataclasses
            field_value = field_values[field_name]
            field_type = resolved_field_types[field_name]
            try:
                field_value_str = field_value._repr_html_(vertical=not vertical)
            except AttributeError:
                field_value_str = repr(field_value)

            table_rows.append(
                f"<tr><td>{field_name}(<i>{field_type}</i>)</td>"
                f"<td>{field_value_str}</td></tr>"
            )
    else:
        table_rows.append(
            "<tr>"
            + "".join(
                [
                    f"<td>{field_name} (<i>{field_type}</i>)</td>"
                    for field_name, field_type in resolved_field_types.items()
                ]
            )
            + "</tr>"
        )
        values = []
        for field_value in field_values.values():
            # Recursively call _repr_html_ for nested dataclasses
            try:
                field_value_str = field_value._repr_html_(
                    vertical=not vertical
                )  # alternates orientation
            except AttributeError:
                field_value_str = repr(field_value)
            values.append(f"<td>{field_value_str}</td>")
        table_rows.append("<tr>" + "".join(values) + "</tr>")
    table_rows.append("</table>")
    return "\n".join(table_rows)


@dataclass_transform(kw_only_default=True)
class MetaDCRegister(type):
    """A MetaClass adding registration for subclasses and transform to dataclass."""

    dunder_name: str

    def __new__(
        meta: type[MetaDCRegister],
        clsname: str,
        bases: tuple,
        class_dict: dict,
    ) -> type:
        """Create a dataclass, with log property and auto registry properties.

        No need for @dataclass decorator.
        """
        class_dict["log"] = property(make_log_property(meta.dunder_name))
        class_dict["_repr_html_"] = dataclass_repr_html
        cls = dataclasses.dataclass(kw_only=True)(
            super().__new__(meta, clsname, bases, class_dict)
        )

        name_lookup = f"__{meta.dunder_name}_name__"
        if getattr(cls, "__registry__", None) is None:
            # Adding a registry to the Base Class.
            cls.__registry__ = {}
        if (r := getattr(cls, name_lookup, None)) is not None:
            cls.__registry__[r] = cls
        return cls


class LogMixin:
    """A Mixin to add a logger to a class."""

    @property
    def log(self) -> logging.Logger:
        """Logger."""
        return logging.getLogger(f"{self.__class__.__name__}")


class NoCaseEnumMeta(EnumMeta):
    """Make Enum case insensitive."""

    def __getitem__(cls, name: Any):
        if isinstance(name, str):
            name = name.upper()
        return super().__getitem__(name)


class NoCaseEnum(Enum, metaclass=EnumMeta):
    """Base Class for Enum to be case insensitive."""

    pass


class MethodRegister:
    """
    A Decorator to register methods of the same type in dictionaries.

    Parameters
    ----------
    name: str
        The  register
    """

    registry: dict = defaultdict(dict)

    def __init__(self, register_name: str):
        self.register_name = register_name

    def __call__(
        self,
        method_name: str | Callable,
    ) -> Callable:
        """Register the function in the registry."""

        def decorator(func: Callable[..., T], method_name: str) -> Callable[..., T]:
            self.registry[self.register_name][method_name] = func
            if func.__name__ != method_name:  # type: ignore
                func.__name__ += "__" + method_name  # type: ignore

            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> T:
                return func(*args, **kwargs)

            return wrapper

        # allow for direct name.
        if callable(method_name):
            func = method_name
            method_name = func.__name__  # type: ignore
            return decorator(func, method_name)
        else:
            return partial(decorator, method_name=method_name)


if sys.version_info <= (3, 12):

    def batched(iterable: Iterable, n: int) -> Iterator[tuple[int]]:
        # batched('ABCDEFG', 3) --> ABC DEF G
        if n < 1:
            raise ValueError("n must be at least one")
        it = iter(iterable)
        while batch := tuple(itertools.islice(it, n)):
            yield batch

else:
    batched = itertools.batched


class Singleton(type):
    _instances: dict[type, object] = {}

    def __call__(cls, *args: Any, **kwargs: Any):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class ENVCONFIG(metaclass=Singleton):
    """Environment Configuration."""

    SNAKE_TMP_DIR = "/tmp"
    SNAKE_HDF5_CHUNK_SIZE = 1024**2
    SNAKE_HDF5_CHUNK_WRITE_SIZE = 4 * 1024**3

    @classmethod
    def __getitem__(cls, key: str) -> Any:
        if key in os.environ:
            return os.environ[key]
        return getattr(cls, key)


EnvConfig = ENVCONFIG()
