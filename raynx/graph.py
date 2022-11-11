from typing import Iterable, Union

import networkx as nx
import yaml
from pydantic import BaseModel, validator

from raynx.models import InputModel
from raynx.node import ConnectedNode, Node
from raynx.ray_utils import ContextModel, RayGlob, RayRemoteOptions


class GraphModel(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    ray: RayRemoteOptions = None
    # This will only hold the root nodes
    # after initialization
    workflow: dict[str, ConnectedNode]
    _nodes: set[ConnectedNode] = None

    @classmethod
    def from_yaml(cls, path):
        return cls(**yaml.safe_load(path))

    @validator("ray")
    def init_ray_glob(cls, val, values, field):
        RayGlob.reinit(val)
        return val

    @validator("workflow", pre=True)
    def init_workflow_graph(cls, val, values, field):
        return _graph_from_config(val)

    def run(self, input_model: InputModel = None, context: ContextModel = None):
        res = [root.compute_as_root(input_model, context) for root in self.workflow.values()]
        for r in res:
            [r_.get() for r_ in r]

    @property
    def nodes(self) -> Iterable[ConnectedNode]:
        _nodes = dict()

        def _add(node: Union[ConnectedNode, Node]):
            _nodes[id(node)] = node
            if isinstance(node, ConnectedNode):
                for cnode in node.to:
                    _add(cnode)

        for node in self.root_nodes:
            _add(node)

        for node in _nodes.values():
            yield node

    @property
    def leaf_nodes(self) -> Iterable[ConnectedNode]:
        for node in self.nodes:
            if not node.to:
                yield node

    @property
    def root_nodes(self):
        for node in self.workflow.values():
            yield node


def _graph_from_config(config_dict):
    graph_config = {key: val.get("to", []) for key, val in config_dict.items()}
    G = nx.DiGraph(graph_config)
    roots = [x for x in G.nodes() if G.in_degree(x) == 0]

    def _initialize(G, config, key):
        if isinstance(G[key], ConnectedNode):
            return G[key]
        to = []
        for n in G[key]:
            to.append(_initialize(G, config, n))
        config_node = config[key]
        config_node["to"] = to
        return ConnectedNode(**config[key])

    initialized_roots = {}
    for root in roots:
        initialized_roots[root] = _initialize(G, config_dict, root)

    return initialized_roots
