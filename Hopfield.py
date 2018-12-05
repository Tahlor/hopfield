import numpy as np

""" Todo: 
        * manage weights
        * parallel processing
        * GPU?
        * optimal weight matrix
"""

class HopfieldNetwork:

    def __init__(self, cost_matrix, values=None):
        self.cost_matrix = cost_matrix
        self.n = cost_matrix.shape[0]
        self.values = np.zeros(*cost_matrix.shape) if values is None else values
        self.nodes = {}
    #     self.build_nodes()
    #
    # def build_nodes(self):
    #     nodes = []
    #     for i,row in enumerate(values):
    #         for j,cell in enumerate(row):
    #             if cell != 0:
    #                 node = {}
    #                 node["value"] = cell
    #     pass

    def construct_weights(self):
        # Construct a TSP set of weights for every node
        # Node = {value:"value", weights_matrix=[], address}

    # def update_node(self, node):
    #     node["value"] = 1 if np.sum(self.values * node["weights"]) > 0 else -1
    #     self.values[node["address"]] = node["value"]
    #     return node["value"]
        pass

    def update_nodes(self):
        #update = np.sum(self.cost_matrix * self.values, axis=1)
        update = np.sum(self.cost_matrix * self.values, axis=1)

        update[update>0]=1
        update[update<=0]=-1
        self.values = update
        return update

    def energy(self):
        return np.sum(self.cost_matrix * self.values)

costs = np.asarray([[0,0,1],[0,0,-1],[1,-1,0]])
values = np.asarray([[-1,-1,-1]]) # not a matrix multiply
#values = np.asarray([[-1,0,0],[0,-1,0],[0,0,-1]])
h = HopfieldNetwork(costs, values)
print(h.update_nodes())
print(h.update_nodes())

h.values = np.asarray([[-1,1,-1]])
print(h.update_nodes())
print(h.update_nodes())
print(h.energy())
