import pytest

from raynx import converter
from raynx.models import InputModel, OutputModel


class SimpleInput(InputModel):

    list_float: list[float]
    single_float: float
    logical: bool


class IterInput(InputModel):
    list_float1: list[float]
    list_float2: list[float]
    list_float_short: list[float]
    single_float: float
    logical: bool


class SimpleOutputCompatible(OutputModel):

    list_float: list[float]
    single_float: float
    logical: bool


class SimpleOutputIncompatible(OutputModel):

    list_float: list[float]
    logical: bool


class SimpleOutputIncompatibleConverted(OutputModel):

    list_float: list[float]
    logical: bool


@converter
def out_to_in(output_model: SimpleOutputIncompatibleConverted) -> SimpleInput:
    return SimpleInput(
        list_float=output_model.list_float, logical=output_model.logical, single_float=2.3
    )


simple_in = SimpleInput(list_float=[1.1, 1.2, -0.2], single_float=2.3, logical=False)
simple_out = SimpleOutputCompatible(list_float=[1.1, 1.2, -0.2], single_float=2.3, logical=False)
simple_out_incomp = SimpleOutputIncompatible(list_float=[1.1, 1.2, -0.2], logical=False)
simple_out_incomp_conv = SimpleOutputIncompatibleConverted(
    list_float=[1.1, 1.2, -0.2], logical=False
)
iter_in = IterInput(
    list_float1=[1.1, 1.2, -0.2],
    list_float2=[2.1, 2.2, -1.2],
    list_float_short=[3.1, 3.2],
    single_float=2.3,
    logical=False,
)


def test_simple_in_out_conversion():
    assert SimpleOutputCompatible.can_convert_to(SimpleInput)
    assert not SimpleOutputIncompatible.can_convert_to(SimpleInput)
    assert simple_out.to_input_model(SimpleInput).dict() == simple_in.dict()
    with pytest.raises(TypeError):
        simple_out_incomp.to_input_model(SimpleInput)

    assert SimpleOutputIncompatibleConverted.can_convert_to(SimpleInput)
    assert simple_out_incomp_conv.to_input_model(SimpleInput).dict() == simple_in.dict()


def test_input_iter():
    assert IterInput.supports_iter(["list_float1", "list_float2", "list_float_short"])
    assert not IterInput.supports_iter(
        ["list_float1", "list_float2", "list_float_short", "logical"]
    )

    model_iter = iter_in.iter_fields("list_float1", batch_size=1)
    assert next(model_iter).list_float1 == [1.1]
    assert next(model_iter).list_float1 == [1.2]
    assert next(model_iter).list_float2 == iter_in.list_float2

    model_iter = iter_in.iter_fields("list_float1", batch_size=2)
    assert next(model_iter).list_float1 == [1.1, 1.2]
    assert next(model_iter).list_float1 == [-0.2]

    model_iter = iter_in.iter_fields(["list_float1", "list_float2"], batch_size=1)
    first = next(model_iter)
    assert first.list_float1 == [1.1]
    assert first.list_float2 == [2.1]
    assert first.logical == iter_in.logical
    second = next(model_iter)
    assert second.list_float1 == [1.2]
    assert second.list_float2 == [2.2]
    assert second.logical == iter_in.logical

    model_iter = iter_in.iter_fields(["list_float1", "list_float2"], batch_size=2)
    first = next(model_iter)
    assert first.list_float1 == [1.1, 1.2]
    assert first.list_float2 == [2.1, 2.2]
    second = next(model_iter)
    assert second.list_float1 == [-0.2]
    assert second.list_float2 == [-1.2]

    model_iter_inner = iter_in.iter_fields(["list_float1"], batch_size=2)
    model_iter_outer = iter_in.iter_fields(["list_float1"], mode="outer", batch_size=2)
    for inner, outer in zip(model_iter_inner, model_iter_outer):
        assert inner.dict() == outer.dict()

    model_iter = iter_in.iter_fields(["list_float1", "list_float2"], mode="outer", batch_size=1)
    first = next(model_iter)
    assert first.list_float1 == [1.1]
    assert first.list_float2 == [2.1]
    second = next(model_iter)
    assert second.list_float1 == [1.1]
    assert second.list_float2 == [2.2]

    model_iter = iter_in.iter_fields(["list_float1", "list_float2"], mode="outer", batch_size=2)
    first = next(model_iter)
    assert first.list_float1 == [1.1, 1.1]
    assert first.list_float2 == [2.1, 2.2]
    second = next(model_iter)
    assert second.list_float1 == [1.1, 1.2]
    assert second.list_float2 == [-1.2, 2.1]

    with pytest.raises(RuntimeError):
        next(iter_in.iter_fields(["list_float1", "list_float2", "list_float_short"], batch_size=2))
