import numpy as np
import matplotlib
import time

""" Todo: 

# Ideas/parameters:
    ### Work on reporting lots of test results at once
    
    # Learning rate
    # Think about simulated annealing?
    # Stochastic updates vs batch
        # Fully random updates
        # Update each node 1x per cycle in random order
        # Batch - update all nodes simultaneously (GPU??)
    # Parallel processing
        # Use mutliple cores
    # Add noise periodically to push out of bad solutions
    # -inf
        # Increase learning -inf through time
        # Set -inf near largest cost in matrix
    # Cost matrix
    # Graphical output:
        # Make a circle corresponding to the magnitude of each guess
    # Early stopping
    
"""

"""
Other:
    # Is there a way we can re-frame the problem to get it work in PyTorch?
    # Maybe our node matrix are our tunable weights
    # Use gradient ascent
    # We're interested in the weights, not the output
    # Our "weights" are the input
    # The output is the temperature
"""

VERBOSE = False


class HopfieldNetwork:

    def __init__(self, cost_matrix, initial_guess=None, inf=-.8, learning_rate=.1, improve_tour_factor=1, clamp_first_column=True):
        """
        Args:
              cost_matrix (2D npy array): Square matrix, expects np.inf for invalid paths
              initial_guess (2D npy array): Initial guess for solutioon
              inf (int): Weight for invalid solution paths; more negative = more invalid
              improve_tour_factor (int): Increasing this should make 1) shorter tours, but 2) increase the liklihood they are invalid
                                            CANNOT be 0, system will yield all 0's since no incentive to travel
        """

        self.inf = inf
        self.n = cost_matrix.shape[0]
        self.improve_tour_factor = improve_tour_factor
        self.clamp_first_column = clamp_first_column
        self.negative_weights = np.ones(self.n - 1) * inf

        if VERBOSE:
            print("Guess:")
            print(self.sol_guess)

        self.original_cost_matrix = cost_matrix.copy()
        self.cost_matrix = self.scale_cost_matrix(cost_matrix)

        self.initialize_guess(initial_guess)

        self.learning_rate = learning_rate

# add a penalty for update

    def initialize_guess(self, guess=None):
        # Start city doesn't matter, we could always start in 0
        if guess is None:
            self.sol_guess = np.random.random([self.n, self.n])

            # Clamp first column
            if self.clamp_first_column:
                self.sol_guess[:,0] = np.zeros(self.n)
                self.sol_guess[0,0] = 1
        else:
            self.sol_guess = guess
        self.report_solution(self.sol_guess)

    def scale_cost_matrix(self, cost_matrix):
        """ Rescale cost matrix, so higher cost routes have smaller weights
            Impossible routes have even smaller weights (e.g. negative)
        """

        # Find which routes are impossible, fill with 0 temporarily
        inf_idx = cost_matrix == np.inf
        cost_matrix[inf_idx] = 0
        _min = np.min(cost_matrix)
        _max = np.max(cost_matrix)

        cost_matrix = 1 - (cost_matrix - _min) / (_max - _min)  # rescale to 0 to 1, then reverse
        # After rescaling, make routes impossible again
        cost_matrix[inf_idx] = self.inf
        return cost_matrix

    def fully_stochastic_update(self, iterations=100):
        lower_bound = 1 if self.clamp_first_column else 0

        for _ in range(0, iterations):
            i = np.random.randint(0, self.n)
            j = np.random.randint(lower_bound, self.n)
            h.update_node(i, j)

        self.report_solution(self.sol_guess)

    def get_path(self, sol):
        # Take argmax
        path = np.argmax(sol, axis=0)

        # If argmax is actually 0, return -1
        column_sum = np.sum(sol, axis=0)
        path[column_sum!=1]=-1
        return path

    def report_solution(self, solution):
        sol = np.round(solution)
        happiness = self.get_happiness(sol)
        path = self.get_path(sol)
        cost = self.get_cost(sol)
        print(solution)
        print("Solution: {}".format(path))
        print("Cost: {}".format(cost))
        print("Valid: {}".format(cost < np.inf))
        print("Happiness: {}".format(happiness))

    def update_node(self, i, j, learning_rate=None, improve_tour_factor=None):
        """ Update a single node
        """
        if improve_tour_factor is None:
            improve_tour_factor = self.improve_tour_factor

        if learning_rate is None:
            learning_rate = self.learning_rate

        # i is an index of cities
        # j is an index of city-visit ordering
        next_city_idx = (j + 1) % self.n  # wrap around to beginning
        indices = np.arange(self.n)

        # Cost matrix - this rewards the system for taking a non-zero path
        update = np.sum(self.sol_guess[:, next_city_idx] * self.cost_matrix[i, :]) * improve_tour_factor # * self.sol_guess[i, j]
        #update = 0
        # print(i,j)
        # print(self.sol_guess)
        # print("solution", self.sol_guess[:, next_city_idx])
        # print(self.cost_matrix)
        # print("cost", self.cost_matrix[i,:])
        # print("update", update)

        # No duplicate visit on column
        update += np.sum(self.sol_guess[indices != i, j] * self.negative_weights)

        # No duplicate visit on row
        update += np.sum(self.sol_guess[i, indices != j] * self.negative_weights)

        # Multiply by original node value; without this, the negative weights want to force everything to zero
        update *= self.sol_guess[i, j]

        # print(update)

        #delta = learning_rate * (1 if update > 0 else -1)  # we can use a tanh here
        delta = learning_rate * np.arctan(update)  # we can use a tanh here

        self.sol_guess[i, j] = max(min(self.sol_guess[i, j] + delta, 1), 0)
        return update

    def get_happiness(self, sol=None, improve_tour_factor=None):
        """ Calculate 'happiness' of network. This is -energy (maximize happiness, minimize energy/entropy)
            Depends on network topology (how we choose to connect nodes)

            Args:
                sol (2d matrix): Our guessed solution
        """
        if improve_tour_factor is None:
            improve_tour_factor = self.improve_tour_factor

        if sol is None:
            sol = self.sol_guess
        indices = np.arange(self.n)
        happiness = 0
        for i in range(0, self.n):
            for j in range(0, self.n):
                next_city_idx = (j + 1) % self.n

                # Costs - this line doesn't seem to be working right
                happiness += sol[i, j] * np.sum(sol[:, next_city_idx] * self.cost_matrix[i, :]) * improve_tour_factor

                # No duplicate visit on column
                happiness += np.sum(sol[indices != i, j] * self.negative_weights) * sol[i, j]

                # No duplicate visit on row
                happiness += np.sum(sol[i, indices != j] * self.negative_weights) * sol[i, j]

        return happiness

    def get_cost(self, solution_matrix):
        """
        Args:
            solution_matrix (array): A list of city indices
        """

        # Make sure solution is valid
        if (np.sum(solution_matrix, axis=1)!=np.ones(self.n)).any() or (np.sum(solution_matrix, axis=0)!=np.ones(self.n)).any():
            return np.inf

        # Compute cost
        cost = 0
        path = np.argmax(solution_matrix,axis=0)
        for i, city_idx in enumerate(path):
            src = path[i]
            dest = path[(i + 1) % self.n]
            cost += self.original_cost_matrix[src, dest]
        return cost

    def plot_current_state(self):
        """ Plot the self.sol_guess matrix, a circle for each matrix cell, radius proportional to magnitude
        """

        pass


if __name__ == "__main__":
    inf = np.inf
    cost_matrix = np.asarray([[inf, 7, 3, 12], [3, inf, 6, 14], [5, 8, inf, 6], [9, 3, 5, inf]])
    n = cost_matrix.shape[0]
    # values = np.asarray([[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]])
    inferior_solution = np.asarray([[1., 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 1, 0, 0]])
    solution = np.asarray([[1., 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]])
    noise = np.random.randn(n,n)
    noise[:,0] = 0
    solution_noise = solution+noise


    guess = solution_noise
    # Solution: 0 2 3 1
    # guess = None

    h = HopfieldNetwork(cost_matrix, initial_guess=guess, improve_tour_factor=.5, learning_rate=.01)
    h.fully_stochastic_update(2000)


