import math

import pydantic
import pytest

from raynx import node
from raynx.models import InputModel, OutputModel, ContextModel
from raynx.node import ConnectedNode
from raynx.ray_options import  RayRemoteOptions


class SimpleInput(InputModel):

    list_float: list[float]
    single_float: float
    logical: bool


class SimpleOutput(OutputModel):

    list_float: list[float]
    single_float: float
    logical: bool


class SimpleOutputIncompatible(OutputModel):

    list_float: list[float]
    logical: bool


class SimpleContext(ContextModel):

    ray: RayRemoteOptions = RayRemoteOptions(use_ray=False)
    batch_size: int = 2


simple_in = SimpleInput(list_float=[1.1, 1.2, -0.2], single_float=2.3, logical=False)
incremented_out = SimpleOutput(list_float=[1.2, 1.3, -0.1], single_float=2.4, logical=True)
incremented_out_chained = SimpleOutput(
    list_float=[1.21, 1.31, -0.09], single_float=2.41, logical=False
)


def increment(input_model, incr=0.1):
    list_float = [round(f + incr, 3) for f in input_model.list_float]
    logical = not input_model.logical
    single_float = round(input_model.single_float + incr, 3)
    return SimpleOutput(list_float=list_float, single_float=single_float, logical=logical)


@node
def increment_all(input_model: SimpleInput, context: SimpleContext = None) -> SimpleOutput:
    return increment(input_model, incr=0.1)


@node
def increment_small(input_model: SimpleInput, context: SimpleContext = None) -> SimpleOutput:
    return increment(input_model, incr=0.01)


def test_wrong_signature():

    with pytest.raises(AttributeError):

        @node
        def wrong_input(input_mode: InputModel) -> SimpleOutput:
            ...

    with pytest.raises(AttributeError):

        @node
        def wrong_context(input_model: InputModel, cont: ContextModel) -> SimpleOutput:
            ...

    with pytest.raises(AttributeError):

        @node
        def missing_input(context: ContextModel) -> SimpleOutput:
            ...

    @node
    def valid(input_model: SimpleInput, context: SimpleContext) -> SimpleOutput:
        ...


def test_missing_annotations():

    with pytest.raises(AttributeError):

        @node
        def missing_input_anno(input_model, context: SimpleContext) -> SimpleOutput:
            ...

    # This is valid but model will need to return None
    @node
    def valid(input_model: SimpleInput, context: SimpleContext):
        ...


def test_node():

    output_model = increment_all.compute(simple_in)
    assert output_model.dict() == incremented_out.dict()


def test_node_provided_type():
    @node(input_type=SimpleInput)
    def increment_all_anno(input_model, context: SimpleContext = None) -> SimpleOutput:
        return increment(input_model)

    output_model = increment_all_anno.compute(simple_in)
    assert output_model.dict() == incremented_out.dict()


def test_node_for_each():
    @node(input_type=SimpleInput, for_each="list_float")
    def increment_all_anno(input_model, context: SimpleContext = None) -> SimpleOutput:
        ...

    @node(input_type=SimpleInput, for_each=["list_float"], for_each_mode="inner")
    def increment_all_anno(input_model, context: SimpleContext = None) -> SimpleOutput:
        ...

    with pytest.raises(ValueError):

        @node(
            input_type=SimpleInput,
            for_each=["list_float", "logical"],
            for_each_mode="inner",
        )
        def increment_all_anno(input_model, context: SimpleContext = None) -> SimpleOutput:
            ...


def test_connect():

    results = []

    def result_hook(output_model):
        results.append(output_model)

    cnode = ConnectedNode(node=increment_small, to=[])
    cnode.add_hook(result_hook)
    connected_increment = increment_all.forward_connect_to(cnode)
    connected_increment.compute(simple_in, context=SimpleContext())
    assert results[-1].dict() == incremented_out_chained.dict()

    @node(input_type=SimpleInput, for_each=["list_float"], for_each_mode="inner")
    def incompatible(input_model, context: SimpleContext = None) -> SimpleOutputIncompatible:
        ...

    with pytest.raises(pydantic.ValidationError):
        incompatible.forward_connect_to(increment_small)

    results.clear()

    @node(input_type=SimpleInput, for_each=["list_float"], for_each_mode="inner")
    def compatible_for_each(input_model, context: SimpleContext = None) -> SimpleOutput:
        return SimpleOutput(**input_model.dict())

    cnode = ConnectedNode(node=increment_small, to=[])
    cnode.add_hook(result_hook)
    connected_increment = increment_all.forward_connect_to(
        compatible_for_each.forward_connect_to(cnode)
    )
    connected_increment.compute(simple_in, context=SimpleContext())
    assert len(results) == math.ceil(len(simple_in.list_float) / SimpleContext().batch_size)
