import numpy as np

""" Todo: 
        * manage weights
        * parallel processing
        * GPU?
        * optimal weight matrix
"""

# Is there a way we can re-frame the problem to get it work in PyTorch?
# Maybe our node matrix are our tunable weights
# Use gradient ascent
# We're interested in the weights, not the output
# Our "weights" are the input
# The output is the temperature

class HopfieldNetwork:

    def __init__(self, cost_matrix, values=None, inf=-1):
        self.inf = inf
        self.n = cost_matrix.shape[0]
        self.values = np.zeros(*cost_matrix.shape) if values is None else values
        self.cost_matrix = self.scale_cost_matrix(cost_matrix)
        self.negative_weights = np.ones(self.n-1)*inf
        self.learning_rate = .1

    def scale_cost_matrix(self, cost_matrix):
        _min = np.min(cost_matrix)
        _max = np.max(cost_matrix)
        np.fill_diagonal(cost_matrix, 0)
        cost_matrix = 1-(cost_matrix - _min)/(_max-_min) # rescale to 0 to 1, then reverse
        np.fill_diagonal(cost_matrix, self.inf)
        return cost_matrix

    def update_node(self,i,j):
        # i is an index of cities
        # j is an index of city-visit ordering
        next_city_idx = (j+1) % self.n
        indices = np.arange(self.n)

        #print(self.values[:,next_city_idx],self.cost_matrix[i,:],self.values[indices!=i,j],self.negative_weights,self.values[i,indices!=j],self.negative_weights)

        # Cost matrix
        update = np.sum(self.values[:,next_city_idx] * self.cost_matrix[i,:])

        # No duplicate visit on column
        update += np.sum(self.values[indices!=i,j]*self.negative_weights)

        # No duplicate visit on row
        update += np.sum(self.values[i,indices!=j]*self.negative_weights)

        delta = self.learning_rate * (1 if update > 0 else -1)

        self.values[i,j] = max(min(self.values[i,j]+delta,1),0)
        return update

    def energy(self):
        return np.sum(self.cost_matrix * self.values)

inf = -1
test = np.asarray([[inf, 7, 3, 12], [3, inf, 6, 14], [5, 8, inf, 6], [9, 3, 5, inf]])
values = np.asarray([[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]])
values = np.asarray([[1.5,0,0,0],[0,0,0,1],[0,1,0,0],[0,0,1,0]])
# Solution: 0 2 3 1

n=test.shape[0]

#values = np.random.random(n,n)
#costs = np.asarray([[0,0,1],[0,0,-1],[1,-1,0]])
#values = np.asarray([[1,1,1],[1,1,1],[1,1,1]])

h = HopfieldNetwork(test, values)
for _ in range(0,100):
    i = np.random.randint(0,n)
    j = np.random.randint(0,n)
    print(i,j)
    h.update_node(i,j)
print(h.values)

