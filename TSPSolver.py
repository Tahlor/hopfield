#!/usr/bin/python3
inf = float('inf')

from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
    from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
    from PyQt4.QtCore import QLineF, QPointF
else:
    raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))

import time
import numpy as np
from TSPClasses import *
from Hopfield import *
import heapq
import itertools
import numpy as np
import sys
import queue as Q
import copy

VERBOSE = False

class TSPSolver:
    def __init__( self, gui_view=None ):
        self._scenario = None
        self.bssf = inf
        self.best_solution = {}
        self.expansions = 0
        self.bssf_updates = 0

    def setupWithScenario( self, scenario ):
        self._scenario = scenario


    def convert_indices_to_cities(self, route):
        cities = self._scenario.getCities()
        return [cities[i] for i in route]

    def build_matrix(self):
        cities = self._scenario.getCities()
        # self._cities = [City(pt.x(), pt.y()) for pt in city_locations]
        cost_matrix = np.zeros([len(cities),len(cities)])

        for i in range(0,len(cities)):
            for j in range(0, len(cities)): # may not be symmetric distances, otherwise use i not 0
                cost_matrix[i,j] = cities[i].costTo(cities[j])
        return cost_matrix

    def reduce(self, cost_matrix, bound=0):
        """ Row and column reduce matrix, so a 0 is found in each. Add the minimum to the bound.
            We create a new n x n matrix O(n^2) space, and modify potentially every cell, O(n^2) time
        """
        # matrix = matrix.copy()
        bound = 0

        # Row
        row_min = np.maximum(np.min(cost_matrix, axis=1), 0)[:, None]
        row_min[row_min == inf] = 0
        bound += np.sum(row_min)
        cost_matrix -= row_min

        # Column
        column_min = np.maximum(np.min(cost_matrix, axis=0), 0)  # [:,None]
        column_min[column_min == inf] = 0
        bound += np.sum(column_min)
        cost_matrix -= column_min
        return cost_matrix, bound

    def update_matrix(self, data, source, destination, offset=True):
        """ Row/column reduce the matrix
            We create a new n x n matrix O(n^2) space, and modify potentially every cell, O(n^2) time
            (this calls reduce, plus does a few extra chores)
        """

        cost_matrix = data["matrix"]
        if offset:
            source -= 1
            destination -= 1
        cost = cost_matrix[source, destination]
        cost_matrix[source, :] = inf
        cost_matrix[:, destination] = inf
        cost_matrix[destination, source] = inf
        cost_matrix, bound = self.reduce(cost_matrix)
        data["bound"] += bound
        data["cost"] += cost + bound
        data["matrix"] = cost_matrix
        self.expansions += 1
        return data

    def main(self, cost_matrix, time_allowance=60, use_greedy=True):
        """ The main algorithm.
            If we didn't do any pruning, traversing our tree would be O(n!) time/space.
            Also, if our pruning is bad, traversing our tree would be O(n!) time/space
            But we don't expand every node; we know we can expand in O(n^2) time/space
            Then if our tree has an average branching factor b, and n levels, we have
                b^n nodes to expand. This is O(n^2 b^n) time. For space, we never need more
                than the bottom of the tree fully expanded. This is will probably still be a constant
                from b^n, so O(n^2 b^n) space.

            Our priority queue is O(n log n) time, O(n) space, which is dwarfed by our tree time/space.
        """
        start = time.time()
        n = cost_matrix.shape[0]
        q = Q.PriorityQueue()

        if use_greedy:
            greedy_results = self.greedy(90, cost_matrix)
            if VERBOSE:
                print("Greedy",greedy_results)
            self.bssf = greedy_results['cost']
            self.best_solution = {"cost": self.bssf, "path": greedy_results['path'], "optimal": ""}

        self.bssf_updates = 0
        self.prune = 0
        self.max_q = 0

        r, bound = self.reduce(cost_matrix)
        data = {"bound": bound, "cost": bound, "path": [0], "n": n, "matrix": r,
                "destinations": list(range(1, n)), "optimal":""}
        q.put((0, data))
        self.expansions = 0

        while not q.empty() and time.time()-start < time_allowance:
            data = q.get()[-1]

            # We're done
            # if self.bssf <= data["bound"]:
            #     self.prune += q.qsize()
            #     #print(q.qsize())
            #     self.best_solution["optimal"]="*"
            #     break
            if self.bssf <= data["bound"]:
                self.prune += 1
                continue

            q = self.recurse_level(data, q)

            # Update maximum size of queue
            self.max_q = q.qsize() if q.qsize()>self.max_q else self.max_q

        if q.empty():
            self.best_solution["optimal"] = "*"
        self.best_solution["max"] = self.max_q
        self.best_solution["total"] = self.expansions
        self.best_solution["bssf_updates"]=self.bssf_updates

        return self.best_solution

    def recurse_level(self, data, q):
        """ Expand the current branch
            The expansion depends on the depth, n-d
            Because the expansion is n^2, this is O(n^2 * n-d) time/space, or O(n^3) for the first layer
        """
        current = (data["path"])[-1]
        depth = data["n"]-len(data["destinations"])
        #print("COST", data["cost"])
        for dest in data["destinations"]:
            d = copy.deepcopy(data)

            # Check each possible city
            d["destinations"].remove(dest)
            d = self.update_matrix(d, current, dest, offset=False)
            d["bound"]=max(d["bound"], d["cost"])

            if data["bound"] > self.bssf:
                self.prune += 1
                continue

            # Update bssf
            if len(data["path"])==data["n"]-1:

                # Check for complete cycle
                origin = data["path"][0]
                last_leg_cost = d["matrix"][dest, origin]
                if last_leg_cost < inf:
                    d["cost"] += last_leg_cost
                    data["path"].append(origin)
                    if VERBOSE:
                        print("New best cost",d["cost"])
                    if d["cost"] < self.bssf:
                        self.bssf = d["cost"]
                        self.best_solution = d
                        self.bssf_updates += 1
                else:
                    if VERBOSE:
                        print("Not a cycle!")

            # Update queue
            d["path"].append(dest)

            # Sort by: best lower bound, depth, cost
            #q.put((-depth, d["bound"], d["cost"], self.expansions, d))
            priority = 2*data["n"]*math.log(d["bound"]) - depth
            q.put((priority, d["bound"], d["cost"], self.expansions, d))

        del data
        return q

    def defaultRandomTour( self, time_allowance=60.0 ):
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        count = 0
        bssf = None
        start_time = time.time()
        while not foundTour and time.time()-start_time < time_allowance:
            # create a random permutation
            perm = np.random.permutation( ncities )
            route = []
            # Now build the route using the random permutation
            for i in range( ncities ):
                route.append( cities[ perm[i] ] )
            bssf = TSPSolution(route)
            count += 1
            if bssf.cost < np.inf:
                # Found a valid route
                foundTour = True
        end_time = time.time()
        results['cost'] = bssf.cost if foundTour else inf
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

    def greedy(self, time_allowance=60.0, cost_matrix=None):
        """ Do a greedy search
            For any particular city, we create the cost matrix, which is O(n**2) time/space
                We then loop through each unvisited city, find a minimum in an array of size n; since we create the matrix once,
                we're still O(n**2) time, O(n**2) space
            In the worst case, we try every possible starting node (if no path is found)
                So O(n**3) time, O(n**2) space
        """
        results = {}
        if cost_matrix is None:
            cities = self._scenario.getCities()
            ncities = len(cities)
            count = 0
            bssf = None
            cost_matrix = self.build_matrix()
        else:
            ncities=cost_matrix.shape[0]

        start_time = time.time()
        no_path_found = True
        unvisited_indices = list(range(0, ncities))
        np.random.shuffle(unvisited_indices)
        while no_path_found and time.time()-start_time < time_allowance and unvisited_indices:
            first = unvisited_indices.pop(0)
            route = [first]
            working_matrix = cost_matrix.copy()
            working_matrix[:,first] = inf
            total_cost = 0
            for u in range(1, ncities):
                source = route[-1]
                min_cost_dest = np.argmin(working_matrix[source])
                cost = working_matrix[source,min_cost_dest]
                if cost == inf:
                    break
                route.append(min_cost_dest)
                total_cost += cost
                working_matrix[:,source] = inf

            # Check last one too
            if len(route)==ncities:
                # Original cost matrix
                final_cost = cost_matrix[route[-1], first]
                if final_cost < inf:
                    no_path_found = False
                    total_cost += final_cost
        solution = TSPSolution(self.convert_indices_to_cities(route))
        end_time = time.time()
        results['cost'] = total_cost if not no_path_found else inf
        results['time'] = end_time - start_time
        results['count'] = None
        results['soln'] = solution
        results['path'] = route
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

    def branchAndBound( self, time_allowance=60.0 ):
        """ This calls main(), see main() for time/space analysis
        """

        results={}
        start_time = time.time()
        cost_matrix = self.build_matrix()

        ## Run algorithm
        data = self.main(cost_matrix, time_allowance=time_allowance)
        end_time = time.time()
        results['optimal'] = data['optimal']
        route=data["path"]
        solution = TSPSolution(self.convert_indices_to_cities(route))

        results['cost'] = str(data['cost'])+data['optimal']
        results['time'] = end_time - start_time
        results['count'] = self.bssf_updates  # number of bssf updates
        results['soln'] = solution
        results['max'] = self.max_q           # states stored at once
        results['total'] = self.expansions    # states created
        results['pruned'] = self.prune        # states skipped
        return results

    def fancy( self,time_allowance=60.0 ):
        startTime = time.time()
        matrix = self.build_matrix()
        cities = self._scenario.getCities()
        network = HopfieldNetwork(matrix, improve_tour_factor=.5, learning_rate=.01)
        results = network.fully_stochastic_update(2000)
        cost = results[1]
        if(cost != math.inf):
            path = results[0]
            listOfCities = []
            for x in path:
                listOfCities.append(cities[x])
            soln = TSPSolution(listOfCities)
            results = {}
            results['cost'] = cost
            results['time'] = time.time() - startTime
            results['count'] = 0
            results['soln'] = soln
            results['max'] = 0
            results['total'] = 0
            results['pruned'] = 0
            return results

        results = {}
        results['cost'] = math.inf
        results['time'] = time.time() - startTime
        results['count'] = 0
        results['soln'] = None
        results['max'] = 0  # states stored at once
        results['total'] = 0  # states created
        results['pruned'] = 0  # states skipped
        return results





if __name__=="__main__":
    x = np.asarray([[inf, 7, 3, 12], [3, inf, 6, 14], [5, 8, inf, 6], [9, 3, 5, inf]])
    solver=TSPSolver()
    solution = solver.main(x)
    print(solution)

# Any state you create counts toward the total.  Any state that doesn't get put in the queue because its bound is larger than the BSSF length counts as pruned, as does any state popped from the queue but then not expanded because its bounds is (now) larger than the BSSF.

