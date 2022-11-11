from raynx import node
from raynx.models import InputModel, OutputModel
from raynx.ray_utils import ContextModel


class FooInput(InputModel):
    val: int


class FooIterOutput(OutputModel):
    vals: list[int]


class BarInput(InputModel):
    vals: list[int]
    prefix: str = "bar_"


class BarOutput(OutputModel):
    message: str


@node
def foo(input_model: FooInput, context: ContextModel = None) -> FooIterOutput:
    new_vals = [input_model.val, input_model.val + 1, input_model.val + 2]
    return FooIterOutput(vals=new_vals)


@node(for_each=["vals"])
def bar(input_model: BarInput, context: ContextModel = None) -> BarOutput:
    val_string = " ".join([str(v) for v in input_model.vals])
    message = f"{input_model.prefix}{val_string}"
    return BarOutput(message=message)
