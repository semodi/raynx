import inspect
from functools import partial
from typing import Callable, Literal, Union

from raynx.models import OutputModel, model_converters
from raynx.node import Node
from raynx.ray_utils import ContextModel


def _validate_signature_node(func: Callable):
    """Ensures that only input_model and context are in function's signature and
    infers and returns types from annotations."""

    ann = func.__annotations__
    InType = ann.get("input_model")
    OutType = ann.get("return")
    ConType = ann.get("context", ContextModel)

    signature = inspect.signature(func).parameters

    if "input_model" not in signature:
        raise AttributeError(f"input_model must be argument of {func.__name__}")

    if "context" not in signature:
        raise AttributeError(f"context must be an argument of {func.__name__}")

    if len(signature) > 2:
        extra = tuple(key for key in signature if key not in ["context", "input_model"])
        raise AttributeError(f"Nodes allow no extra arguments but found {extra}")

    return InType, OutType, ConType


def _validate_signature_converter(func: Callable):
    """Ensures that only input_model and context are in function's signature and
    infers and returns types from annotations."""

    ann = func.__annotations__
    InType = ann.get("output_model")
    OutType = ann.get("return")

    signature = inspect.signature(func).parameters

    if "output_model" not in signature:
        raise AttributeError(f"output_model must be argument of {func.__name__}")

    if len(signature) > 2:
        extra = tuple(key for key in signature if key not in ["context", "input_model"])
        raise AttributeError(f"Nodes allow no extra arguments but found {extra}")

    return InType, OutType


def _node_decorator(
    func: Callable,
    input_type: type = None,
    output_type: type = None,
    context_type: type = None,
    name: str = None,
    for_each: Union[str, list[str]] = None,
    for_each_mode: Literal["inner", "outer"] = "inner",
):

    InType, OutType, ConType = _validate_signature_node(func)
    input_type = input_type or InType

    # Allow for None as OutputModel if data sink (leaf node)
    output_type = output_type or OutType or OutputModel
    context_type = context_type or ConType

    def err(which_type):
        raise AttributeError(f"{which_type} type could not be determined for {func.__name__}")

    if not input_type:
        err("input")
    if not output_type:
        err("output")
    if not context_type:
        err("context")

    if for_each and not input_type.supports_iter(for_each):
        raise ValueError(f"for_each {for_each} not a field of {input_type}")

    if for_each_mode not in ["inner", "outer"]:
        raise ValueError(f"for_each_mode {for_each_mode} invalid")

    fields = {}
    fields["_input_type"] = property(lambda _: input_type)
    fields["_output_type"] = property(lambda _: output_type)
    fields["for_each"] = property(lambda _: for_each)
    fields["for_each_model"] = property(lambda _: for_each_mode)
    if context_type:
        fields["_context_type"] = property(lambda _: context_type)
    fields["_wrapped_callable"] = staticmethod(func)
    fields["name"] = name or func.__name__

    return type(f"Node({fields['name']})", (Node,), fields)()


def _converter_decorator(
    func: Callable,
    input_type: type = None,
    output_type: type = None,
):

    OutType, InType = _validate_signature_converter(func)
    input_type = input_type or InType
    output_type = output_type or OutType

    def err(which_type):
        raise AttributeError(f"{which_type} type could not be determined for {func.__name__}")

    if not input_type:
        err("input")
    if not output_type:
        err("output")

    model_converters[output_type][input_type] = func


def node(*args, **kwargs):
    """Node decorator to transform a function into a raynx node.
    Function must have signature:
    ``foo(input_model: InputModel, context: ContextModel) -> OutputModel``

    Returns:
        Type
            A node class
    Example:

        .. code-block:: python

            @node
            def foo(input_model: FooInput, context: FooContext) -> FooOutput:
                ...
                return output_model

            @node(input_type = FooInput, name='my_bar')
            def bar(input_model, context: BarContext) -> BarOutput:
                ...
                return output_model

    """
    if len(args) == 1 and callable(args[0]):
        return _node_decorator(args[0], **kwargs)
    else:
        return partial(_node_decorator, **kwargs)


def converter(*args, **kwargs):
    """Decorator to register a converter.
    Function must have signature:
    ``foo(output_model: OutputModel) -> InputModel``

    Example:

        .. code-block:: python

            @converter
            def foo_converter(output_model: FooOutput) -> BarInput:
                ...
                return input_model 

    """
    if len(args) == 1 and callable(args[0]):
        return _converter_decorator(args[0], **kwargs)
    else:
        return partial(_converter_decorator, **kwargs)
