from TSPSolver import *
from TSPClasses import *
from PyQt5.QtCore import *
from Hopfield import *
#from PyQt5.QtWidgets import *
#from PyQt5.QtGui import *
import pandas as pd

# GLobals
MAX_TIME = 600
MAX_TIME_FANCY = 900
AUTO = False
SIZE = 100
EASY=0
NORMAL=1
HARD=2
HARDD=3
DIFFICULTY = HARDD
#rand_seed = 10

class TSP_Problem():
    def __init__(self):
        SCALE=1
        self.data_range = {'x': [-1.5 * SCALE, 1.5 * SCALE],
                           'y': [-SCALE, SCALE]}

        #self.solver = TSPSolver(self.view)

    def generateNetwork_nogui(self, diff, rand_seed, size):
        points = self.newPoints(rand_seed, size)  # uses current rand seed

        diffs = {0: "Easy", 1: "Normal", 2: "Hard", 3: "Hard (Deterministic)"}
        diff = diffs[diff]
        self._scenario = None
        self._scenario = Scenario(city_locations=points, difficulty=diff, rand_seed=rand_seed)
        self.genParams = {'size': str(size), 'seed': str(rand_seed), 'diff': str(diff)}
        return self._scenario

    def newPoints(self, seed, npoints):
        random.seed(seed)
        ptlist = []
        RANGE = self.data_range
        xr = self.data_range['x']
        yr = self.data_range['y']
        while len(ptlist) < npoints:
            x = random.uniform(0.0, 1.0)
            y = random.uniform(0.0, 1.0)
            if True:
                xval = xr[0] + (xr[1] - xr[0]) * x
                yval = yr[0] + (yr[1] - yr[0]) * y
                ptlist.append(QPointF(xval, yval))
        return ptlist

def test(SIZE, rand_seed):
    w = TSP_Problem()
    solver = TSPSolver()

    # Generate scenario
    scenario = w.generateNetwork_nogui(DIFFICULTY, rand_seed, SIZE)
    solver.setupWithScenario(scenario)

    # Branch and bound
    results = solver.branchAndBound(time_allowance=MAX_TIME)
    print("BranchBound", SIZE, rand_seed, results["time"], results["cost"], results["max"], results["count"], results["total"], results["pruned"])
    best_cost = float(results["cost"].replace("*",""))
    bb_route = results["route"]

    # Greedy
    results = solver.greedy(time_allowance=MAX_TIME)
    print("Greedy", SIZE, rand_seed, results["time"], results["cost"], results["max"], results["count"], results["total"], results["pruned"])
    greedy_cost = results["cost"]
    if results["path"] != []:
        one_hot = utils.one_hot(results["path"])
    else:
        one_hot = np.random.random([SIZE,SIZE])
    #one_hot = utils.one_hot(bb_route)


    # Random
    results=[]
    start = time.time()
    for i in range(0,100):
    #while time.time()-start < MAX_TIME:
        results.append(solver.smartRandomTour(time_allowance=MAX_TIME))
    results = pd.DataFrame(results)
    total_time = results["time"].sum()
    total_iterations = len(results)
    results = results.replace([np.inf, -np.inf], np.nan)
    avg_cost = results[["cost"]].dropna(axis=0).mean()[0]
    best_random_cost = results[["cost"]].dropna(axis=0).min()[0]
    print("Random", SIZE, rand_seed, total_time, avg_cost, best_random_cost, total_iterations)

    # Fancy
    cost_matrix = solver.build_matrix()
    # network = HopfieldNetwork(cost_matrix, improve_tour_factor=.85, learning_rate=.3, inhibition_factor=1,
    #                   force_visit_bias=.0, epochs=150, optimal_cost=best_cost, when_to_force_valid=1.1,
    #                   force_valid_factor=4, clamp_first_column=True, cost_matrix_exponent=1)

    # Greedy initialization
    # network = HopfieldNetwork(cost_matrix, improve_tour_factor=.85, learning_rate=.06, inhibition_factor=1.1,
    #                   force_visit_bias=.0, epochs=200, optimal_cost=best_cost, when_to_force_valid=.75,
    #                   force_valid_factor=4, clamp_first_column=False, cost_matrix_exponent=1, global_inhibition_factor=1)
    # results = solver.fancy(time_allowance=MAX_TIME_FANCY, network=network, simulations=100, guess=one_hot,
    #                        run_until_optimal=True)
    # print(SIZE, rand_seed, results["time"], results["cost"], results["max"], results["count"], results["total"],
    #       results["pruned"])

    # GOOD
    # network = HopfieldNetwork(cost_matrix, improve_tour_factor=.85, learning_rate=.3, inhibition_factor=1.1,
    #                   force_visit_bias=.0, epochs=200, optimal_cost=best_cost, when_to_force_valid=.75,
    #                   force_valid_factor=4, clamp_first_column=False, cost_matrix_exponent=1, global_inhibition_factor=1)

    improve_tour_factors = {60:.75, 100:.65, 200:.5}
    if size >= 60:
        improve_tour_factor = improve_tour_factors[size]
    else:
        improve_tour_factor = .85

    network = HopfieldNetwork(cost_matrix, improve_tour_factor=improve_tour_factor, learning_rate=.3, inhibition_factor=1.07,
                              force_visit_bias=.0, epochs=250, optimal_cost=0, when_to_force_valid=.65,
                              force_valid_factor=4, clamp_first_column=True, cost_matrix_exponent=1,
                              global_inhibition_factor=1, anneal=False)

    # Annealing stuff
    # network = HopfieldNetwork(cost_matrix, improve_tour_factor=.85, learning_rate=.1, inhibition_factor=1.07,
    #                           force_visit_bias=.0, epochs=500, optimal_cost=best_cost, when_to_force_valid=.65,
    #                           force_valid_factor=4, clamp_first_column=True, cost_matrix_exponent=1,
    #                           global_inhibition_factor=1, anneal=True)


    # Greedy Fancy
    #results = solver.fancyGreedy(time_allowance=MAX_TIME_FANCY, network=network, greedy_size=2)
    #print("FancyGreedy", SIZE, rand_seed, results["time"], results["cost"], results["max"], results["count"], results["total"],
    #      results["pruned"])

    #network.update_cost_matrix(cost_matrix)
    
    network.optimal_cost=best_cost
    if True:
        results = solver.fancy(time_allowance=MAX_TIME_FANCY, network=network, simulations=500, guess=None, run_until_optimal=True)
        print("Fancy", SIZE, rand_seed, results["time"], results["cost"], results["max"], results["count"], results["total"], results["pruned"])
    else:
        network.make_movie(one_hot)
    # print(one_hot)
    # print(cost_matrix)
    del scenario

if __name__=="__main__":
    #print("Cities",	"Seed",	"Running Time",	"Cost of best tour found",	"Max # of stored states at a given time	", "# of BSSF updates",	"Total # of states created",	"Total # of states pruned")
    #for repititions in range(0, 5):
    #for size in [30,60,100,200]:
    size = 200
    rand_seed = np.random.randint(0, 1000)
    test(size, rand_seed)

# Good solutions:
# 5 348

## GOOD FOR 20: no annealing, single side weights
