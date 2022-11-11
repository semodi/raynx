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
