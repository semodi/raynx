from collections import defaultdict
from collections.abc import Iterable
from typing import Literal, Union
from warnings import warn

from pydantic import BaseModel

from raynx.utils import batch, iter_modes
from raynx.ray_options import RayRemoteOptions, RayGlob

model_converters = defaultdict(dict)


def _default_converter(model):
    def convert(data):
        return model(**data.dict())

    return convert


class DataModel(BaseModel):
    class Config:
        arbitrary_types_allowed = True
        allow_mutation = False
        extra = "ignore"


class InputModel(DataModel):
    @classmethod
    def supports_iter(cls, fields: Union[str, list[str]] = None) -> bool:
        """Whether fields are contained in this InputModel and can be iterated over"""
        if isinstance(fields, str):
            fields = [fields]
        for field in fields:
            try:
                # For complex fields like list[float]
                field_type = cls.__fields__[field].outer_type_.__origin__
            except AttributeError:
                field_type = cls.__fields__[field].type_
                warn(
                    f"Iterating over non-complex field of type {field_type}, "
                    "is that really what you are trying to do?"
                )

            if field not in cls.__fields__ or not issubclass(field_type, Iterable):
                return False
        return True

    def iter_fields(
        self,
        fields: Union[str, list[str]] = None,
        mode: Literal["outer", "inner"] = "inner",
        batch_size: int = 1,
    ) -> "InputModel":
        """Yield copies of this InputModel with specified fields replaced by
        iterated values.

        Args:
            fields (Union[str, list[str]], optional): Fields to iterate over. Defaults to None.
            mode (Literal["outer", "inner"], optional): How multiple fields are to be combined
                when iterating over them. Inner product vs. Outer product. Defaults to "inner".
            batch_size (int, optional): The batch size. Defaults to 1

        Raises:
            AttributeError: If fields cannot be iterated over (because not contained
                or not iterable)

        Yields:
            Iterator[InputModel]: Copy of self with selected fields iterated over
        """
        if not fields:
            yield self
            return

        if isinstance(fields, str):
            fields = [fields]

        if not self.supports_iter(fields):
            raise AttributeError(f"{type(self)} cannot iterate over {fields}")

        iterator = iter_modes[mode]([getattr(self, f) for f in fields])
        for field_batch in batch(iterator, batch_size):
            dict_ = self.dict()
            for field_name, field_value in zip(fields, zip(*field_batch)):
                dict_.update({field_name: field_value})

            yield type(self)(**dict_)


class OutputModel(DataModel):
    def to_input_model(self, input_type: type) -> InputModel:
        if not issubclass(input_type, InputModel):
            raise TypeError(f"{input_type} not a subclass of InputModel")

        if not type(self).can_convert_to(input_type):
            raise TypeError(
                f"OutputModel subclass {type(self)} model cannot be converted to {input_type}."
            )

        converter = model_converters.get(type(self), {}).get(
            input_type, _default_converter(input_type)
        )
        return converter(self)

    @classmethod
    def can_convert_to(cls, input_type: type) -> bool:
        if cls == OutputModel:
            raise TypeError(
                "OutputModel base model cannot be converted to input model."
                " Please subclass OutputModel"
            )
        if cls in model_converters and input_type in model_converters[cls]:
            return True
        else:
            # TODO: This is a rather weak assumption
            required_fields = {
                key for key, field in input_type.__fields__.items() if field.required
            }
            if not required_fields.difference(set(cls.__fields__.keys())):
                warn(f"Will use default converter to convert from {cls} to {input_type}")
                return True
            else:
                return False

class ContextModel(DataModel):

    ray: RayRemoteOptions = None
    batch_size: int = 1

    @property
    def use_ray(self):
        return self.ray.use_ray or RayGlob.use_ray