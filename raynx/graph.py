from typing import Iterable, Union

import networkx as nx
import yaml
from pydantic import BaseModel, validator

from raynx.models import InputModel, ContextModel
from raynx.node import ConnectedNode, Node
from raynx.ray_utils import RayGlob
from raynx.ray_options import RayRemoteOptions


class GraphModel(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    ray: RayRemoteOptions = None
    # This will only hold the root nodes
    # after initialization
    workflow: dict[str, ConnectedNode]

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
        """Iterates over all nodes in graph"""
        nodes_ = dict()

        def add(node: Union[ConnectedNode, Node]):
            nodes_[id(node)] = node
            if isinstance(node, ConnectedNode):
                for cnode in node.to:
                    add(cnode)

        for node in self.root_nodes:
            add(node)

        for node in nodes_.values():
            yield node

    @property
    def leaf_nodes(self) -> Iterable[ConnectedNode]:
        """Iterates over all leaf nodes in graph"""
        for node in self.nodes:
            if not node.to:
                yield node

    @property
    def root_nodes(self):
        """Iterates over all root nodes in graph"""
        for node in self.workflow.values():
            yield node


def _graph_from_config(config_dict):
    """Initialize a computational graph from a config dictionary

    Args:
        config_dict (dict): config dictionary describing the graph 
            structure 

    Returns:
        dict[str, ConnectedNode]: The connected root nodes of the graph

    Example:
        A config dictionary would have the following (minimal) structure.
        Nodes without a ``to`` section are considered terminal or leaf nodes.

        .. code-block:: 

            config_dict = {
                mynode1: {
                    node: node_callable1 
                    to:
                        - mynode2 
                        - mynode3
                },
                mynode2: {
                    node: node_callable2
                    to:
                        ...
                },
                mynode3: {
                    node: node_callable2 #The same callable can be re-used at multiple places in the graph
                    # No ``to`` section, hence a mynode3 is leaf 
                }
            }

        
    """
    # Extract relevant information to infer graph connectivity  ("to" section)
    graph_config = {key: val.get("to", []) for key, val in config_dict.items()}

    # Build a networkx DAG and identify root nodes from connectivity
    G = nx.DiGraph(graph_config)
    roots = [x for x in G.nodes() if G.in_degree(x) == 0]

    # Recursive initialization: Connects nodes to each other. 
    # The recursion is called on the root nodes but effectively connects the 
    # graph backwards starting from the leaf nodes. 
    initialized = {}
    def initialize(G, config, key):
        # To avoid creating duplicate ConnectedNodes
        # return the cached node if it exists
        if key in initialized:
            return initialized[key]

        # Recursion: Initialize all the nodes connected to this one
        to = []
        for next_ in G[key]:
            to.append(initialize(G, config, next_))
        config_node = config[key]
        config_node["to"] = to
        initialized[key] = ConnectedNode(**config[key])
        return initialized[key]

    # As all nodes are either connected and/or root nodes we only
    # need to keep references to the root nodes to access the entire graph
    initialized_roots = {}
    for root in roots:
        initialized_roots[root] = initialize(G, config_dict, root)

    return initialized_roots
