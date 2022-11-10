from abc import ABC, abstractmethod, abstractproperty
from collections import defaultdict
from copy import deepcopy
from typing import Any, Callable, Optional, Union
from warnings import warn

from pydantic import BaseModel, validator

from raynx.models import InputModel, OutputModel
from raynx.ray_utils import ContextModel, RayFuture, RaySwitchRemote


class Node(ABC):

    _instances = {}

    def __init__(self):
        """A node represents an operation in a computational graph.
        Every node takes an input_model of type ``_input_type`` and returns
        an output_model of type ``_output_type``. Computational requirements
        and other fixed options can be specified using a context_model of type
        ``_context_type``
        """
        if self.name in self._instances:
            warn(
                f"An instance with a class name {self.name} already exists. Overwriting..."
            )
        self._instances[self.name] = self

    @classmethod
    def get_instance_by_name(cls, name: str):
        """Retrieves an in instance of a Node subclass
        by using the subclass's name. Assumes that there
        is only one instance per subclass.

        Args:
            name (str): If Node was created by decorating a function,
                this name will be equal to the function's name, unless
                otherwise specified.

        Returns:
            Node:
        """
        return cls._instances[name]

    @property
    def to(self):
        return []

    @property
    def for_each(self):
        return None

    @property
    def for_each_mode(self):
        return "inner"

    @abstractproperty
    def _input_type(self):
        """Type of input_model"""
        ...

    @abstractproperty
    def _output_type(self):
        """Type of output_model (return type of ``self.compute()``)"""
        ...

    @property
    def _context_type(self):
        """Type of context model"""
        return ContextModel

    @staticmethod
    @abstractmethod
    def _wrapped_callable(input_model: InputModel, context: ContextModel):
        """This should be overwritten with the decorated function."""
        raise NotImplementedError()

    def __call__(self):
        raise RuntimeError(
            "Nodes should not be called directly, use node.compute() instead"
        )

    def compute(
        self, input_model: InputModel, context: Optional[ContextModel] = None
    ) -> OutputModel:
        """Call the wrapped callable and validate types of input, context and
            output models.

        Args:
            input_model (InputModel): Model containing input data that is to be used
                for computation by the ``Node``
            context (Optional[ContextModel], optional): Context variables, e.g.
                computational requirements. Defaults to None.

        Raises:
            TypeError: If any of the models (input, context, output) has an unexpected
                type.


        Returns:
            OutputModel:
        """
        if not isinstance(input_model, self._input_type):
            raise TypeError(f"input_model {type(input_model)} not of type {self._input_type}")
        if context and not isinstance(context, self._context_type):
            raise TypeError(f"context {type(context)} not of type {self._context_type}")
        res = self._wrapped_callable(input_model, context)
        res = res or OutputModel()
        if not isinstance(res, self._output_type):
            raise TypeError(f"output_model {type(res)} not of type {self._output_type}")

        return res

    def forward_connect_to(
        self, connected_nodes: list["Node"], **kwargs
    ) -> "ConnectedNode":
        """Connect this node to ``connected_nodes``. The output model of this
        Node will be sent to connected_nodes as input (after appropriate conversion).

        Args:
            connected_nodes (list[Node]): Nodes to connect to

        Returns:
            ConnectedNode: self as a ConnectedNode with ``connected_nodes`` as ``to``
                attribute.
        """
        if not isinstance(connected_nodes, list):
            connected_nodes = [connected_nodes]
        return ConnectedNode(node=self, to=connected_nodes, **kwargs)

    def runtime_context(self, context: ContextModel):
        return context


class ConnectedNodeBase(BaseModel):
    """Base model for a connected Node."""

    class Config:
        arbitrary_types_allowed = True

    def __getattribute__(self, __name: str) -> Any:
        try:
            return super().__getattribute__(__name)
        except AttributeError:
            return getattr(self.node, __name)


class ConnectedNode(ConnectedNodeBase):
    """BaseModel that at, its core, contains a Node that is
    connected to other Nodes.

    Args:
        node (Node, str): Can be initialized by providing a Node, or
            the name of a Node
        to (list[Union[Node, ConnectedNodeBase, str]]): references
            to nodes that this ConnectedNode sends data to. Can be
            initialized by providing the names of Nodes.
        kwargs (Optional[dict]): kwargs dict that should overwrite
            fields in input_model before ``node.compute`` is called.
        context (Optional[dict]): context dict that should overwrite
            fields in context_model before ``node.compute`` is called.
    """

    node: Node
    to: list[Union[Node, ConnectedNodeBase]]
    kwargs: Optional[dict] = None
    context: Optional[dict] = None
    output_hooks: list[Callable] = []

    @validator("node", pre=True)
    def validate_node(cls, val, values, field):
        """Convert to Node if string (name) encountered"""
        if isinstance(val, str):
            val = Node.get_instance_by_name(val)
        return val

    @validator("to", each_item=True, pre=True)
    def validate_to_nodes(cls, val, values, field):
        """Convert to Node if string (name) encountered,
        Ensure that this node's output model can be converted to
        the connected node's input model"""
        if isinstance(val, str):
            val = Node.get_instance_by_name(val)
        ot = values["node"]._output_type
        it = val._input_type
        if not ot.can_convert_to(it):
            raise TypeError(f"Nodes incompatible, {ot} cannot convert to {it}")
        return val

    @validator("kwargs", pre=True)
    def validate_kwargs(cls, val, values, field):
        """Makes sure that kwargs can overwrite the nodes input model."""
        node = values["node"]
        node_fields = node._input_type.__fields__
        for key, v in val.items():
            if key not in node_fields:
                raise KeyError(f"{key} not a valid {node._input_type} field")
            if not isinstance(v, node_fields[key].type_):
                val[key] = node_fields[key].type_(v)
        return val

    @validator("context", pre=True)
    def validate_context(cls, val, values, field):
        """Makes sure that context can overwrite the node's context model."""
        node = values["node"]
        node_fields = node._context_type.__fields__
        for key, v in val.items():
            if key not in node_fields:
                raise KeyError(f"{key} not a valid {node._context_type} field")
            if not isinstance(v, node_fields[key].type_):
                try:
                    val[key] = node_fields[key].type_(v)
                except TypeError:
                    if isinstance(v, dict):
                        val[key] = node_fields[key].type_(**v)
                    else:
                        raise
        return val

    def add_hook(self, hook: Callable):
        self.output_hooks.append(hook)
    
    def clear_hooks(self):
        self.output_hooks.clear()

    def runtime_context(self, context: ContextModel = None):
        # Use self.context to overwrite fields in passed context
        context_dict = {}
        if context:
            context_dict = deepcopy(context.dict())
        if self.context:
            context_dict.update(self.context)
        if context_dict:
            context = self.node._context_type(**context_dict)
            return context
        return self.node._context_type()

    def compute_as_root(
        self, input_model: InputModel, context: ContextModel = None
    ) -> dict[str, OutputModel]:
        runtime_context = self.runtime_context(context)
        results = []
        for im in input_model.iter_fields(
            self.node.for_each,
            mode=self.node.for_each_mode,
            batch_size=runtime_context.batch_size,
        ):
            results.append(RaySwitchRemote(self.compute, context=runtime_context).options(name=self.node.name).remote(im, context))

        return results

    def compute(
        self, input_model: InputModel, context: ContextModel = None
    ) -> dict[str, OutputModel]:
        """Run compute of self.node (optionally using ray).

        Args:
            input_model (InputModel): Model containing input data that is to be used
                for computation by the ``self.node``
            context (Optional[ContextModel], optional): Context variables, e.g.
                computational requirements. Defaults to None.

        Returns:
            dict[str, OutputModel]:
        """

        # Use self.kwargs to overwrite fields in input_model
        if self.kwargs:
            input_dict = input_model.dict()
            input_dict.update(self.kwargs)
            input_model = self.node._input_type(**input_dict)

        # Call compute myself and wait for results
        output_model = self.node.compute(input_model, self.runtime_context(context))
        node_results = defaultdict(list)
        node_results["result"] = output_model

        # Convert my output model to input models and send to connected nodes
        for cnode in self.to:
            input_model = output_model.to_input_model(cnode._input_type)
            cnode_context = cnode.runtime_context(context)
            # Iterate over field that is indicated by cnode.for_each (if None, single iteration)
            for im in input_model.iter_fields(
                cnode.for_each,
                mode=cnode.for_each_mode,
                batch_size=cnode_context.batch_size,
            ):
                cnode_output: RayFuture = (
                    RaySwitchRemote(cnode.compute, context=cnode_context)
                    .options(name=cnode.name)
                    .remote(im, context)
                )
                node_results[cnode.name].append(cnode_output)

        # Retrieve results from futures (blocking)
        for cnode in self.to:
            for res in node_results[cnode.name]:
                res.get()

        if self.output_hooks:
            for hook in self.output_hooks:
                hook(output_model)
