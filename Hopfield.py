import numpy as np


class HopfieldNetwork:

    def __init__(self, cost_matrix, values):
        self.cost_matrix = cost_matrix
        self.n = cost_matrix.shape[0]
        self.values = np.zeros(*cost_matrix.shape)
        self.nodes = {}
        self.build_nodes()

    def build_nodes(self):
        nodes = []
        for i,row in enumerate(values):
            for j,cell in enumerate(row):
                if cell != 0:
                    node = {}
                    node["value"] = cell


    def construct_weights(self):
        # Construct a TSP set of weights for every node
        # Node = {value:"value", weights_matrix=[], address}

    def update_node(self, node):
        node["value"] = 1 if np.sum(self.values * node["weights"]) > 0 else -1
        self.values[node["address"]] = node["value"]
        return node["value"]



costs = np.asarray([[0,0,1],[0,0,-1],[1,-1,0]])
values = np.asarray([[-1,0,0],[0,-1,0],[0,0,-1]])
h = HopfieldNetwork(costs, values)
