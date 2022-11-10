from raynx.graph import GraphModel
from raynx.models import InputModel, OutputModel
from raynx.ray_utils import ContextModel, RayRemoteOptions
from raynx import node

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
incremented_out = SimpleOutput(
    list_float=[1.2, 1.3, -0.1], single_float=2.4, logical=True
)
incremented_out_chained = SimpleOutput(
    list_float=[1.21, 1.31, -0.09], single_float=2.41, logical=False
)

results = []

def result_hook(output_model):
    results.append(output_model)

def increment(input_model, incr=0.1):
    list_float = [round(f + incr, 3) for f in input_model.list_float]
    logical = not input_model.logical
    single_float = round(input_model.single_float + incr, 3)
    return SimpleOutput(
        list_float=list_float, single_float=single_float, logical=logical
    )

@node(name="increment_LARGE")
def increment_all(
    input_model: SimpleInput, context: SimpleContext = None
) -> SimpleOutput:
    return increment(input_model, incr=0.1)


@node(name='increment_SMALL')
def increment_small(
    input_model: SimpleInput, context: SimpleContext = None
) -> SimpleOutput:
    return increment(input_model, incr=0.01)

@node(name='increment_SMALL_foreach', for_each=['list_float'])
def increment_small_foreach(
    input_model: SimpleInput, context: SimpleContext = None
) -> SimpleOutput:
    return increment(input_model, incr=0.01)

simple_graph = """
ray:
    use_ray: False
workflow:
    increment_by_0.1:
        node: increment_LARGE
        to:
            - increment_by_0.01
    increment_by_0.01:
        node: increment_SMALL
"""

simple_graph_foreach= """
ray:
    use_ray: False
workflow:
    increment_by_0.1:
        node: increment_LARGE
        to:
            - increment_by_0.01
    increment_by_0.01:
        node: increment_SMALL
        to:
            - increment_by_0.01_batched
    increment_by_0.01_batched:
        node: increment_SMALL_foreach
"""

simple_graph_reuse = """
ray:
    use_ray: False
workflow:
    increment_by_0.1:
        node: increment_LARGE
        to:
            - increment_by_0.01
    increment_by_0.01:
        node: increment_SMALL
        to:
            - increment_by_0.1_2
    increment_by_0.1_2:
        node: increment_LARGE
"""

simple_graph_kwargs = """
ray:
    use_ray: False
workflow:
    increment_by_0.1:
        node: increment_LARGE
        to:
            - increment_by_0.01
        kwargs:
            logical: True
    increment_by_0.01:
        node: increment_SMALL
"""

simple_graph_foreach_context= """
ray:
    use_ray: False
    batch_size: 2
workflow:
    increment_by_0.1:
        node: increment_LARGE
        to:
            - increment_by_0.01
    increment_by_0.01:
        node: increment_SMALL
        to:
            - increment_by_0.01_batched
    increment_by_0.01_batched:
        node: increment_SMALL_foreach
        context:
            batch_size: 3
"""

def test_simple_graph():
    graph = GraphModel.from_yaml(simple_graph)
    
    assert set(node.name for node in graph.root_nodes) == {'increment_LARGE'}
    assert set(node.name for node in graph.leaf_nodes) == {'increment_SMALL'}
    for leaf_node in graph.leaf_nodes:
        leaf_node.add_hook(result_hook)
    
    graph.run(simple_in, SimpleContext())
    assert results[-1].dict() == incremented_out_chained.dict()

def test_simple_graph_foreach():
    graph = GraphModel.from_yaml(simple_graph_foreach)
    
    assert set(node.name for node in graph.root_nodes) == {'increment_LARGE'}
    assert set(node.name for node in graph.leaf_nodes) == {'increment_SMALL_foreach'}
    for leaf_node in graph.leaf_nodes:
        leaf_node.add_hook(result_hook)
    results.clear()
    graph.run(simple_in, SimpleContext())
    assert results[0].dict() == {"list_float": [1.22, 1.32],
                                 "single_float": 2.42,
                                 "logical": True}


def test_simple_graph_reuse():
    graph = GraphModel.from_yaml(simple_graph_reuse)
    
    assert set(node.name for node in graph.root_nodes) == {'increment_LARGE'}
    assert set(node.name for node in graph.leaf_nodes) == {'increment_LARGE'}
    for leaf_node in graph.leaf_nodes:
        leaf_node.add_hook(result_hook)

    results.clear() 
    graph.run(simple_in, SimpleContext())
    expected_out = SimpleOutput(
        list_float=[1.31, 1.41, 0.01], single_float=2.51, logical=True
    )
    assert results[-1].dict() == expected_out.dict()

def test_simple_graph_kwargs():
    graph = GraphModel.from_yaml(simple_graph_kwargs)
    
    assert set(node.name for node in graph.root_nodes) == {'increment_LARGE'}
    assert set(node.name for node in graph.leaf_nodes) == {'increment_SMALL'}
    for leaf_node in graph.leaf_nodes:
        leaf_node.add_hook(result_hook)

    results.clear() 
    graph.run(simple_in, SimpleContext())
    expected_out = SimpleOutput(
        list_float=[1.21, 1.31, -0.09], single_float=2.41, logical=True
    )
    assert results[-1].dict() == expected_out.dict()

def test_simple_graph_foreach():
    graph = GraphModel.from_yaml(simple_graph_foreach_context)
    
    assert set(node.name for node in graph.root_nodes) == {'increment_LARGE'}
    assert set(node.name for node in graph.leaf_nodes) == {'increment_SMALL_foreach'}
    for leaf_node in graph.leaf_nodes:
        leaf_node.add_hook(result_hook)
    results.clear()
    graph.run(simple_in, SimpleContext())
    assert results[0].dict() == {"list_float": [1.22, 1.32, -0.08],
                                 "single_float": 2.42,
                                 "logical": True}