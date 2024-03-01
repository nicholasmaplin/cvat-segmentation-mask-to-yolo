from typing import Generic, List, TypeVar

import numpy as np
from pydantic import BaseModel
from pydantic.fields import ModelField
from typing_extensions import Literal

JSON_ENCODERS = {np.ndarray: lambda arr: arr.tolist()}

DType = TypeVar("DType")


class Coordinates2D(np.ndarray, Generic[DType]):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, val, field: ModelField):
        if not isinstance(val, np.ndarray):
            raise TypeError("numpy.ndarrray required")

        if val.shape[1] != 2:
            raise TypeError(
                (
                    "must be (N x 2) array, where N is the number of points "
                    f"got: {val.shape}"
                )
            )

        dtype_field = field.sub_fields[0]
        actual_dtype = dtype_field.type_.__args__[0]
        # If numpy cannot create an array with the request dtype, an error will be raised
        # and correctly bubbled up.
        np_array = np.array(val, dtype=actual_dtype)
        return np_array


class BaseModelPolygon(BaseModel):  # pylint: disable=too-few-public-methods
    coordinates: Coordinates2D[Literal["int32"]]


class BaseModelMultiPoly(BaseModel):  # pylint: disable=too-few-public-methods
    coordinates: List[Coordinates2D[Literal["int32"]]]
