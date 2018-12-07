from TSPSolver import *
from TSPClasses import *
from Hopfield import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

# GLobals
MAX_TIME = 8
AUTO = False
SIZE = 5
EASY=0
NORMAL=1
HARD=2
HARDD=3
DIFFICULTY = HARD

class TSP_Problem():
    def __init__(self):
        SCALE=1
        self.data_range = {'x': [-1.5 * SCALE, 1.5 * SCALE], \
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

def test():
    w = TSP_Problem()
    solver = TSPSolver()
    print("Cities",	"Seed",	"Running Time",	"Cost of best tour found",	"Max # of stored states at a given time	", "# of BSSF updates",	"Total # of states created",	"Total # of states pruned")

    # Generate scenario
    rand_seed = np.random.randint(0, 1000)
    scenario = w.generateNetwork_nogui(DIFFICULTY, rand_seed, SIZE)
    solver.setupWithScenario(scenario)

    # Branch and bound
    results = solver.branchAndBound(time_allowance=MAX_TIME)
    print(SIZE, rand_seed, results["time"], results["cost"], results["max"], results["count"], results["total"], results["pruned"])
    best_cost = float(results["cost"].replace("*",""))

    # Greedy
    results = solver.greedy(time_allowance=MAX_TIME)
    print(SIZE, rand_seed, results["time"], results["cost"], results["max"], results["count"], results["total"], results["pruned"])

    # Fancy
    cost_matrix = solver.build_matrix()
    network = HopfieldNetwork(cost_matrix, initial_guess=None, improve_tour_factor=1.7, learning_rate=1,
                      force_visit_bias=0, epochs=120, optimal_cost=best_cost, when_to_force_valid=.75, force_valid_factor=10)
    results = solver.fancy(time_allowance=MAX_TIME, network=network, simulations=1000)
    print(SIZE, rand_seed, results["time"], results["cost"], results["max"], results["count"], results["total"], results["pruned"])

    del scenario

if __name__=="__main__":
    test()

# Good solutions:
# 5 348