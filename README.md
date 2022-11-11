# raynx

Ray + pydantic + networkx = Type-validated DAG workflows.

# Installation

The only hard dependencies are 

- ray
- pydantic
- networkx

ray_graph can be installed with pip:

```pip install .```

# Example

For a self-contained example please see `examples/` 

# Concepts

## The computational graph

A workflow should be expressed as an directed acyclic graph (DAG). Nodes of this graph correspond to specific stand-alone computations
that can be chained together to build more complex (distributed) workflows. 

## Data Models

Raynx supports three basic data models that should be subclassed: `InputModel`, `OutputModel`, and `ContextModel`. 
Each node of the computational path should accept an `InputModel` and optionally a `ContextModel` as input 
and return an `OutputModel`. `InputModel` should hold all data used by the the computation whereas `ContextModel` 
can be used to specify computational resources and fine-tune the behavior of Ray. As all three models are subclassing
pydantic `BaseModel`, everything is type-validated automatically.


## Nodes

Nodes correspond to actions/computations and should be provided as functions that take an `InputModel` and optionally a `ContextModel` 
as input, and return an OutputModel.
To indicate that a function will be a node in a graph it has to be decorated with `@node`
like so:

```python 
class FooInput(InputModel):
    val: int

class FooOutput(OutputModel):
    val: int

@node 
def foo(input_model: FooInput, context: ContextModel = None) -> FooOutput:
  new_val = input_model.val + 1
  return FooOutput(val=new_val)
```

Instead of using type annotations, all data models can also be specified in the decorator:

```python
@node(input_type = FooInput, context_type = ContextModel, output_type = FooOutput)
def foo(input_model, context = None):
  new_val = input_model.val + 1
  return FooOutput(val=new_val)
```
Nodes are automatically registered with the workflow graph using the functions name (``foo``), if a different name 
is to be used it can be specified in the decorator:

```python
@node(name="my_foo")
def foo(...):
    ...
```

A node can iterate over an attribute of an InputModel using the keyword `for_each`:

```python 
class FooIterInput(InputModel):
    vals: list[float]
    vals2: list[float]

class FooOutput(OutputModel):
    val: int

@node(for_each = "vals")
def foo(input_model: FooIterInput, context: ContextModel = None) -> FooOutput:
  new_val = input_model.vals[0] + 1
  return FooOutput(val=new_val)
```

Mutliple attributes can be iterated over at the same time, using either an inner product or an 
outer product between the selected attributes. 

```python

@node(for_each = ["vals", "vals2"], for_each_model="inner") # 'inner' is the default mode
def foo_inner(input_model: FooIterInput, context: ContextModel = None) -> FooOutput:
    ...

# assuming vals = [1.1, 1.2, 1.3], vals2 = [2.1, 2.2, 2.3]
# foo_inner would iterate over
#    [1.1, 2.1]
#    [1.2, 2.2]
#    [1.3, 2.3]

@node(for_each = ["vals", "vals2"], for_each_model="outer")
def foo_outer(input_model: FooIterInput, context: ContextModel = None) -> FooOutput:
    ...

# assuming vals = [1.1, 1.2, 1.3], vals2 = [2.1, 2.2, 2.3]
# foo_inner would iterate over
#    [1.1, 2.1]
#    [1.1, 2.2]
#    [1.1, 2.3]
#    [1.2, 2.1]
#     ...
```

## Configuration 

The most straightforward way to define and configure the workflow is through a `.yaml` file.
This file should contain at least a `workflow` section, outlining the graph:

`workflow.py`
```python 
from raynx.models import InputModel, OutputModel
from raynx.ray_utils import ContextModel
from raynx import node

class FooInput(InputModel):
    val: int

class FooIterOutput(OutputModel):
    vals: list[int]

class BarInput(InputModel):
    vals: list[int]
    prefix: str = 'bar_'

class BarOutput(OutputModel):
    message: str

@node 
def foo(input_model: FooInput, context: ContextModel = None) -> FooIterOutput:
    new_vals = [input_model.val, input_model.val + 1, input_model.val + 2]
    return FooIterOutput(vals=new_vals)

@node(for_each=['vals'])
def bar(input_model: BarInput, context: ContextModel = None) -> BarOutput:
    val_string = ' '.join([str(v) for v in input_model.vals])
    message = f'{input_model.prefix}{val_string}'
    return BarOutput(message=message)
```

`workflow.yaml`
```yaml
ray: # Ray can be configured globally 
   use_ray: False
   num_cpus: 2
   num_gpus: 0
workflow:
  add_one: # These names can be chosen arbitrarily
     node: foo # This name has to match the name the node is registered by
     to:
       - make_message_1  # Send the output to this node
       - make_message_2  # Send the output to this node
     context: # Node-specific context can be set here, this will overwrite any context_model provided to the function directly
        ray: # Ray can also be configured on a per-node level (overwrites global options)
          use_ray: False
  make_message_1:
     node: bar
     kwargs: # kwargs can be used to set input model attributes to a fixed value:
        prefix: "make_message_1: "
  make_message_2:
     node: bar # Nodes can be used more than once 
     kwargs:
        prefix: "make_message_2: "
     context:
        batch_size: 2 # For nodes that set for_each, a batch size can be specified.
                      # By default, the batch size is 1.
```

`run.py`
```python
from workflow import *
from raynx import GraphModel
graph = GraphModel.from_yaml(open("workflow.yaml"))

# Node outputs are not retained by default (we assume that any data that needs to be 
# retained is written to disk by nodes), but a hook can be attached to any node which 
# will be called on the output model. 
# We can use it to store the outputs to a 'global' container:
results = []
def result_hook(result: BarOutput):
    results.append(result.message)

for leaf_node in graph.leaf_nodes:
    leaf_node.add_hook(result_hook)

input_model = FooInput(val=1)
graph.run(input_model)

for result in results:
    print(result)
>>> "make_message_1: 1"
>>> "make_message_1: 2"
>>> "make_message_1: 3"
>>> "make_message_2: 1 2"
>>> "make_message_2: 3"
# The order of these results can be arbitrary
```

