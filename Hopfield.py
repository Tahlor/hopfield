import numpy as np
import matplotlib
import time
import pandas as pd
import collections
import logging
import utils
import multiprocessing
import os
import matplotlib.pyplot as plt

## Things to check:
##  inf distances factor
##  annealing
##  bidirectional
""" Todo: 

# Ideas/parameters:

    # Have greedy find good subproblems, feed into hopfield network
    # Start with Greedy and add noise
    # Build in some checker to see if done relaxing
    # Tune parameters

    # Make fully connected - make connections between non-neighboring cities
    `   # Length is minimal distance to other city
        # Decays for longer 

    # Tune parameters:
        # After 50 iterations, if invalid, increase appropriate bias
    # Use a CNN+LSTM
    # Learning rate
    # Think about simulated annealing?
    # Stochastic updates vs batch
        # Fully random updates
        # Update each node 1x per cycle in random order
        # Batch - update all nodes simultaneously (GPU??)
    # Add noise periodically to push out of bad solutions
    # -inf
        # Increase learning -inf through time
        # Set -inf near largest cost in matrix
    # Cost matrix - alternatives
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

# GLOBALS
inf = np.inf
poolcount = multiprocessing.cpu_count()
#poolcount=1

## Logging
logger = logging.getLogger(__name__)
logger.setLevel("WARNING")
#logger.setLevel("INFO")
logging.getLogger().addHandler(logging.StreamHandler())

class HopfieldNetwork:

    def __init__(self, cost_matrix, inhibition_factor=1.0, epochs=100, learning_rate=.1, improve_tour_factor=1.0,
                 force_visit_bias=0.0, clamp_first_column=True, optimal_cost=inf, when_to_force_valid=.75, force_valid_factor=1.0,
                 cost_matrix_exponent=1.0, global_inhibition_factor=1.0):
        """
        Args:
              cost_matrix (2D npy array): Square matrix, expects np.inf for invalid paths
              initial_guess (2D npy array): Initial guess for solution
              inhibition_factor (int): Weight for invalid solution paths; more negative = more invalid
              epochs: how many times each node will undergo relaxation update
              improve_tour_factor (int): Increasing this should make 1) shorter tours, but 2) increase the liklihood they are invalid
                                            CANNOT be 0, system will yield all 0's since no incentive to travel
              force_visit_bias (int): Positive increases the liklihood a city is visited;
                                    If not enough cities are visited, this promotes ANY visit (not skewed toward optimal ones)
              clamp_first_column (bool): If true, we start with city 0, do not update
              optimal_cost (int): Best known cost for this problem
              when_to_force_valid (float): after what percent of epochs will we force validity
              force_valid_factor (float): multiply these constraints by this amount
        """

        # globals
        self.inhibition_factor = inhibition_factor
        self.n = cost_matrix.shape[0]
        self.improve_tour_factor = improve_tour_factor
        self.clamp_first_column = clamp_first_column
        self.negative_weights = -np.ones(self.n - 1)
        self.force_visit_bias = force_visit_bias
        self.optimal_cost = optimal_cost
        self.when_to_force_valid = when_to_force_valid
        self.force_valid_factor = force_valid_factor
        self.cost_matrix_exponent = cost_matrix_exponent
        self.original_cost_matrix = cost_matrix.copy()
        self.global_inhibition_factor = global_inhibition_factor

        self.cost_matrix = self.scale_cost_matrix(cost_matrix.copy())

        #logger.debug("Guess: \n {}".format(self.sol_guess))

        self.learning_rate = learning_rate
        self.epochs = epochs

# add a penalty for update??

    def initialize_guess(self, guess=None):
        np.random.seed(None)
        # Start city doesn't matter, we could always start in 0
        if guess is None:
            guess = np.random.random([self.n, self.n])

            # Clamp first column
            if self.clamp_first_column:
                guess[:,0] = np.zeros(self.n)
                guess[0,0] = 1
        return guess

    def scale_cost_matrix(self, cost_matrix):
        """ Rescale cost matrix, so higher cost routes have smaller weights
            Impossible routes have even smaller weights (e.g. negative)
        """

        # Find which routes are impossible, fill with 0 temporarily
        inf_idx = cost_matrix == np.inf
        cost_matrix[inf_idx] = 0
        temp_cost_matrix = cost_matrix**self.cost_matrix_exponent
        _min = np.min(temp_cost_matrix)
        _max = np.max(temp_cost_matrix)

        cost_matrix = 1 - (temp_cost_matrix - _min) / (_max - _min)  # rescale to 0 to 1, then reverse

        # After rescaling, make routes impossible again
        cost_matrix[inf_idx] = -abs(self.inhibition_factor)*2
        return cost_matrix

    def fully_stochastic_update(self, iterations=None, sol_guess=None):
        if iterations is None:
            iterations = self.epochs * self.n ** 2
        lower_bound = 1 if self.clamp_first_column else 0

        if sol_guess is None:
            sol_guess = self.initialize_guess()

        for _ in range(0, iterations):
            i = np.random.randint(0, self.n)
            j = np.random.randint(lower_bound, self.n)
            self.update_node(i, j)
        return self.report_solution(sol_guess)

    def shock_out_of_invalid(self, sol_guess, verbose=False):
        if verbose:
            print(sol_guess)
        sum_rows = np.where(sol_guess.sum(axis=1) > 1.4)[0]
        sum_cols = np.where(sol_guess.sum(axis=0) > 1.4)[0]
        sum_rows0 = np.where(sol_guess.sum(axis=1) < .5)[0]
        sum_cols0 = np.where(sol_guess.sum(axis=0) < .5)[0]

        # print(sum_cols, sum_rows)
        # pairs = utils.cartesian_product(sum_cols, sum_rows)
        if len(sum_cols):
            for j in sum_cols:
                for i in range(0, self.n):
                    sol_guess[i, j] /= 3
        if len(sum_rows):
            for i in sum_rows:
                for j in range(0, self.n):
                    sol_guess[i, j] /= 3

        if len(sum_cols0):
            for j in sum_cols0:
                for i in range(0, self.n):
                    sol_guess[i, j] += .4
        if len(sum_rows0):
            for i in sum_rows0:
                for j in range(0, self.n):
                    sol_guess[i, j] += .4
        if verbose:
            print(sol_guess)
        return sol_guess

    def balanced_stochastic_update(self, iterations=None, keep_states=False, sol_guess=None):
        if not iterations is None:
            epochs = np.ceil(iterations / self.n ** 2)
        else:
            epochs = self.epochs
        if sol_guess is None:
            sol_guess = self.initialize_guess()
        else:
            sol_guess = sol_guess.copy()
        learning_rate = self.learning_rate
        when_to_force_valid = self.when_to_force_valid
        force_valid_factor = self.force_valid_factor
        improve_tour_factor = self.improve_tour_factor
        inhibition_factor = self.inhibition_factor
        force_visit_bias = self.force_visit_bias
        global_inhibition_factor = self.global_inhibition_factor
        n = self.n
        indices = np.arange(n)
        cost_matrix = self.cost_matrix.copy()
        lower_bound = 1 if self.clamp_first_column else 0
        i_s = range(0, self.n)
        j_s = range(lower_bound, self.n)
        all_pairs = utils.cartesian_product(i_s, j_s)

        # Keep track of states
        if keep_states:
            self.states = []

        for e in range(0,int(epochs)):
            # Randomize order
            np.random.shuffle(all_pairs)
            VERBOSE = False

            # Force solution to be valid
            if e > epochs * when_to_force_valid:
                if True:
                    if VERBOSE:
                        pass
                        #print("Epoch {}".format(e))
                    #print(":",improve_tour_factor, inhibition_factor)
                    # too_many_columns.shuffle()
                    # too_few_columns.shuffle()
                    # too_many_rows.shuffle()

                    # Do this once:
                    if e-1 <= epochs * when_to_force_valid:
                        #print("CHECK")
                        col = sol_guess.sum(axis=1)
                        row = sol_guess.sum(axis=0)
                        # print(col)
                        # print(row)
                        if (row<.1).any() or (row>1.5).any() or (col<.1).any() or (col > 1.5).any():
                            #print("CLEAR")
                            sol_guess=self.shock_out_of_invalid(sol_guess)


                    too_few_columns, too_many_columns, too_few_rows, too_many_rows = self.local_update_tour_factor(sol_guess)

                    for i in too_few_rows:
                        for j in too_few_columns:
                            if VERBOSE:
                                print("too few", i, j)
                            update = self.calculate_update(i, j, n, sol_guess, cost_matrix, improve_tour_factor*force_valid_factor,
                                                           inhibition_factor, global_inhibition_factor,
                                                           force_visit_bias, indices)
                            self.update_node(i, j, sol_guess, update, learning_rate)
                    for i in too_many_rows:
                        for j in too_many_columns:
                            if VERBOSE:
                                print("too many", i, j)
                            update = self.calculate_update(i, j, n, sol_guess, cost_matrix, improve_tour_factor,
                                                           inhibition_factor*force_valid_factor, global_inhibition_factor,
                                                           force_visit_bias, indices)
                            self.update_node(i, j, sol_guess, update, learning_rate)
                #improve_tour_factor, inhibition_factor= self.global_update_tour_factor(sol_guess)
                #print(improve_tour_factor, inhibition_factor)

            # Random order updates
            temp = 1 / ((e+1) * 1 / epochs)/2

            for pair in all_pairs:
                i = pair[0]
                j = pair[1]
                update = self.calculate_update(i, j, n, sol_guess, cost_matrix, improve_tour_factor, inhibition_factor, global_inhibition_factor, force_visit_bias, indices)
                #if e > epochs*.3:
                if True:
                   sol_guess = self.update_node(i,j, sol_guess, update, learning_rate)
                else:
                   sol_guess = self.annealing_update(i,j, sol_guess, update, learning_rate, temperature=temp)

            if keep_states:
                self.states.append(sol_guess.copy())
        return self.report_solution(sol_guess)

    def global_update_tour_factor(self, sol_guess):
        path = self.get_path(np.round(sol_guess))
        #pathT = self.get_path(np.round(sol_guess.transpose()))
        improve_tour_factor = self.improve_tour_factor
        inhibition_factor = self.inhibition_factor

        if -1 in path and -2 not in path: # not enough city visits
            improve_tour_factor = improve_tour_factor * self.force_valid_factor
            print("Not enough city visits")
        elif -2 in path and -1 not in path: # too many city visits
            #improve_tour_factor = self.improve_tour_factor / self.force_valid_factor
            inhibition_factor = inhibition_factor * self.force_valid_factor
            print("Too many city visits")

        # elif -2 in pathT and -1 not in pathT: # multiple city visits
        #     inhibition_factor = inhibition_factor * self.force_valid_factor
        #     print("City visited multiple times")
        # elif -1 in pathT and -2 not in pathT: # city unvisited
        #     inhibition_factor = improve_tour_factor * self.force_valid_factor
        #     print("City not visited")

        return improve_tour_factor, inhibition_factor


    def local_update_tour_factor(self, sol_guess):
        path = self.get_path(np.round(sol_guess))
        pathT = self.get_path(np.round(sol_guess.transpose()))
        too_few_columns = np.where(path == -1)[0]
        too_many_columns = np.where(path == -2)[0]
        too_few_rows = np.where(pathT == -1)[0]
        too_many_rows = np.where(pathT == -2)[0]
        return too_few_columns, too_many_columns, too_few_rows, too_many_rows


    def get_path(self, sol=None):
        # Take argmax
        path = np.argmax(sol, axis=0)

        # If argmax is actually 0, return -1
        column_sum = np.sum(sol, axis=0)
        path[column_sum>1]=-2 # means we visited too many cities
        path[column_sum<1]=-1 # means we visited 0 cities
        return path

    def report_solution(self, solution):
        sol = np.round(solution)
        happiness = self.get_happiness(sol)
        path = self.get_path(sol)
        cost = self.get_cost(sol)
        logger.debug(solution)
        logger.debug("solution: {}".format(path))
        logger.debug("cost: {}".format(cost))
        logger.debug("valid: {}".format(cost < np.inf))
        logger.debug("happiness: {}".format(happiness))

        counts = collections.Counter(path)
        result = {'cost': cost,
             'valid': cost < np.inf,
             'optimal': cost==self.optimal_cost,
             'path':path,
             'too few cities': counts[-1],
             'too many cities': counts[-2],
             'completion_score': self.completion_score(sol_guess=solution)
                  }

        logger.info(result)
        return result

    def calculate_update(self, i, j, n, sol_guess, cost_matrix, improve_tour_factor, inhibition_factor, global_inhibition_factor, force_visit_bias, indices):

        # i is an index of cities
        # j is an index of city-visit ordering
        next_city_idx = (j + 1) % n  # wrap around to beginning
        previous_city_idx = (j - 1) % n

        # A positive value promotes visits
        update = force_visit_bias * improve_tour_factor * n

        # Cost matrix - this rewards the system for taking a non-zero path to the next city
        #update += np.sum(sol_guess[:, next_city_idx] * cost_matrix[i, :]) * improve_tour_factor

        # Rewards system for previous cities
        update += np.sum(sol_guess[:, previous_city_idx] * cost_matrix[:, i]) * improve_tour_factor * 2

        # Global inhibition - neg if too many
        g = (n-np.sum(sol_guess)) * global_inhibition_factor # / self.n
        update += g

        # No duplicate visit on column
        update += np.sum(sol_guess[indices != i, j] * -1) * abs(inhibition_factor)

        # No duplicate visit on row
        update += np.sum(sol_guess[i, indices != j] * -1) * abs(inhibition_factor)

        return update

    def update_node(self, i, j, sol_guess, update, learning_rate):
        """ Update a single node
        """
        delta = np.sign(update) * learning_rate
        sol_guess[i, j] = max(min(sol_guess[i, j] + delta, 1), 0)
        return sol_guess

    def annealing_update(self, i, j, sol_guess, update, learning_rate, temperature=1):
        """ Use NEGATIVE temperature with happiness
        """
        temperature = abs(temperature)
        p1 = round(1 / (1 + np.exp(-(update) / temperature)),4)
        step_direction = np.random.random() <= p1
        #
        # if step_direction != round(p1):
        #     print("WRONG WAY")
        #     print(step_direction, round(p1))

        delta = (step_direction*2-1) * learning_rate
        sol_guess[i, j] = max(min(sol_guess[i, j] + delta, 1), 0)
        return sol_guess

    def get_happiness(self, sol, improve_tour_factor=None):
        """ Calculate 'happiness' of network. This is -energy (maximize happiness, minimize energy/entropy)
            Depends on network topology (how we choose to connect nodes)

            Args:
                sol (2d matrix): Our guessed solution
        """
        if improve_tour_factor is None:
            improve_tour_factor = self.improve_tour_factor

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

    def get_cost(self, solution_matrix=None):
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

    def plot_current_state(self, state):
        """ Plot the sol_guess matrix, a circle for each matrix cell, radius proportional to magnitude
        """
        r = range(0, self.n)
        size=200
        pairs = utils.cartesian_product(r, r)
        pairs = np.transpose(pairs, (1, 0))
        plt.scatter(pairs[0],pairs[1], s=state.reshape(-1) ** 2 * size, color="blue")
        plt.grid(which="major")
        plt.xticks(np.arange(0, self.n))
        plt.yticks(np.arange(0, self.n))
        #plt.draw()
        plt.show()

    def run_simulation(self, sol_guess=None):
        #print(sol_guess)
        if isinstance(sol_guess, int):
            sol_guess=None
        return self.balanced_stochastic_update(sol_guess=sol_guess)


    def run_simulations(self, simulations=100):
        results = {}
        pool = multiprocessing.Pool(processes=poolcount)
        start = time.time()
        for i in range(0,simulations-1):
            results[i] = pool.apply_async(self.run_simulation)
        pool.close()
        pool.join()
        results = [r.get() for r in results.values()]

        end = time.time()

        best_result = results[np.argmin([r['cost'] for r in results])]
        avg_results = self.summary(results)

        best_result["time"] = end-start
        best_result["attempts"] = simulations
        return best_result, avg_results

    def make_movie(self, guess=None):
        if guess is None:
            guess = self.initialize_guess()
        result = self.balanced_stochastic_update(keep_states=True, sol_guess=guess)
        states = np.asarray(self.states)
        utils.create_movie(data=states, path=r"./movie.mp4", plt_func=self.plot_current_state)
        print(result["cost"])

    def run_until_optimal(self, max_time=60, update_method="balanced_stochastic_update", max_tries=500, guess=None, parallel=True):
        found_optimal = False
        start = time.time()
        results = []
        # Dispatch subprocesses
        pool = multiprocessing.Pool(processes=poolcount)
        #func = eval("self.{}".format(update_method))
        #func = self.balanced_stochastic_update
        func = self.run_simulation
        APPLY_ASYNC=False

        def check_result(result):
            global found_optimal, total_attempts
            #print("DONE")
            out_of_time=False
            if result["cost"] <= self.optimal_cost and self.optimal_cost<inf:
                print("Found optimal")
                found_optimal=True
                if result["cost"] < self.optimal_cost:
                    logger.warn("Provided optimal value was not optimal.")
                pool.terminate()
            elif time.time() - start > max_time:
                #print("Out of time")
                #time.sleep(4)
                pool.terminate()
                out_of_time=True
            if APPLY_ASYNC:
                results.append(result)
            return out_of_time

        if parallel:
            if not APPLY_ASYNC:
                #temp_results = pool.imap_unordered(func,max_tries*[guess])
                temp_results = pool.imap_unordered(func, range(0,max_tries))
                pool.close()

                # Loop through results as they come in
                for result in temp_results:
                    # self.plot_current_state()
                    results.append(result)
                    out_of_time = check_result(result)
                    if out_of_time:
                        break
            else:
                for i in range(0, max_tries):
                    pool.apply_async(func=self.run_simulation, args=[guess], callback=check_result)
                pool.close()
                pool.join()
        else:
            results = []
            for i in range(0, max_tries):
                result = self.run_simulation(guess)
                results.append(result)
                b = check_result(result)
                if b:
                    break

        total_attempts = len(results)
        end = time.time()

        if found_optimal:
            best_result = result
        else:
            # get minimum cost result
            best_result = results[np.argmin([r['cost'] for r in results])]
        best_result["time"] = end-start
        best_result["attempts"] = total_attempts
        return best_result, self.summary(results)

    def completion_score(self, sol_guess):
        # Higher means the further we are from convergence
        return abs(self.n - np.sum(sol_guess[sol_guess>.5]))/self.n

    def summary(self, results):
        # Create summary
        df = pd.DataFrame(results)
        df = df.replace([np.inf, -np.inf], np.nan)
        avg_cost = df[["cost"]].dropna(axis=0).mean()
        avg_everything = df.drop(["cost"], axis=1).mean()
        avg_everything = avg_everything.append(avg_cost)
        print(avg_everything)
        return avg_everything

def toy_problem():
    cost_matrix = np.asarray([[inf, 7, 3, 12], [3, inf, 6, 14], [5, 8, inf, 6], [9, 3, 5, inf]])
    n = cost_matrix.shape[0]
    # values = np.asarray([[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]])
    inferior_solution = np.asarray([[1., 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 1, 0, 0]])
    solution = np.asarray([[1., 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]])
    noise = np.random.randn(n,n)
    noise[:,0] = 0
    solution_noise = solution+noise

    guess = None
    # Solution: 0 2 3 1
    # guess = None

    h = HopfieldNetwork(cost_matrix, improve_tour_factor=.5, learning_rate=.01, force_visit_bias=.5)
    h.fully_stochastic_update(2000)

if __name__ == "__main__":
    #toy_problem()

    #cost_matrix = [[inf, 7, 3, 12], [3, inf, 6, 14], [5, 8, inf, 6], [9, 3, 5, inf]]
    _cost_matrix = ([[inf, 884.0, 836.0, 875.0, 1444.0, 578.0, 329.0, 1203.0, 1293.0],
     [884.0, inf, 332.0, 1719.0, 832.0, 1156.0, 691.0, 1086.0, 2084.0],
     [836.0, 332.0, inf, 1571.0, 1163.0, 934.0, 548.0, 1371.0, 2116.0],
     [875.0, 1719.0, 1571.0, inf, 2315.0, 699.0, 1038.0, 1979.0, 1122.0],
     [1444.0, 832.0, 1163.0, 2315.0, inf, 1892.0, 1404.0, 700.0, 2302.0],
     [578.0, 1156.0, 934.0, 699.0, 1892.0, inf, 489.0, 1763.0, 1597.0],
     [329.0, 691.0, 548.0, 1038.0, 1404.0, 489.0, inf, 1327.0, 1616.0],
     [1203.0, 1086.0, 1371.0, 1979.0, 700.0, 1763.0, 1327.0, inf, 1695.0],
     [1293.0, 2084.0, 2116.0, 1122.0, 2302.0, 1597.0, 1616.0, 1695.0, inf]])
    _cost_matrix = np.asarray(_cost_matrix)

    h = HopfieldNetwork(_cost_matrix, improve_tour_factor=.9, learning_rate=.3, inhibition_factor=1,
                      force_visit_bias=0, epochs=60, optimal_cost=15, when_to_force_valid=.75,
                        force_valid_factor=3,clamp_first_column=False, cost_matrix_exponent=1)
    #h.run_simulations()
    #h.run_until_optimal()
    h.make_movie()