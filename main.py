import math
import time

import numpy as np
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from functools import partial
import matplotlib.pyplot as plt

np.random.seed(11)


# sagatavo problēmas datus
def get_vrp():
    vrp = {}
    vrp['depot'] = 0
    vrp['trucks'] = 15
    vrp['stops'] = 50
    vrp['coords'] = np.random.randint(0, 10000, (vrp['stops'], 2))  # koordinātas ir metros
    vrp['trash'] = np.random.randint(1, 10, vrp['stops'])  # kubikmetri atkritumu
    vrp['trash'][vrp['depot']] = 0  # depo nav atkritumu
    vrp['capacities'] = 20  # mašīnu ietilpība

    tmp = []
    tmp.append((0, 1440))
    for _ in range(1, vrp['stops']):
        start = np.random.randint(0, 72) * 10  # laiks ir minūtēs
        end = min(1440, start + np.random.randint(1, 72) * 10)
        tmp.append((start, end))

    vrp['time_windows'] = tmp

    vrp['trash_time'] = 5  # minūtes, ko aizņem iztukšot vienu miskasti
    vrp['speed'] = 600  # mašīnas ātrums, ~40 km/h uz m/min

    return vrp


# atgriež attālumu starp diviem punktiem, izmanto Eiklīda distanci
def euclidean_distance(x, y):
    return int(math.dist(x, y))


# distances novērtētājs, uztaisa arī dictionary ar distancēm, lai var ātri visam piekļūt
def create_distance_evaluator(vrp):
    distances = {}
    for i in range(vrp['stops']):
        distances[i] = {}
        for j in range(vrp['stops']):
            if i == j:
                distances[i][j] = 0
            else:
                distances[i][j] = (euclidean_distance(vrp['coords'][i], vrp['coords'][j]))

    def distance_evaluator(manager, i, j):
        return distances[manager.IndexToNode(i)][manager.IndexToNode(j)]

    return distance_evaluator


# atkritumu apjomu novērtētājs, apjomam katrā apstāšanās vietā
def create_demand_evaluator(vrp):
    demands = vrp['trash']

    def demand_evaluator(manager, node):
        return demands[manager.IndexToNode(node)]

    return demand_evaluator


# mašīnu ietilpīguma novērtētājs, pievieno constraint par maksimālo apjomu
def add_capacity_constraints(routing, vrp, demand_evaluator_index):
    routing.AddDimension(
        demand_evaluator_index,
        0,  # slack time šeit nav
        vrp['capacities'],
        True,  # no sākuma mašīna ir tukša
        'Capacity')


# laika novērtētājs
def create_time_evaluator(vrp):

    # novērtē, cik ilgi būš apkalpot apstāšanās vietu
    def service_time(vrp, node):
        return vrp['trash'][node] * vrp['trash_time']

    # novēŗtē, cik ilgi jābrauc no viena punkta uz nākamo
    def travel_time(vrp, i, j):
        if i == j:
            travel_time = 0
        else:
            travel_time = euclidean_distance(vrp['coords'][i], vrp['coords'][j]) / vrp['speed']
        return travel_time

    # uztaisa dictionary ar laikiem starp punktiem (ieskaitot, cik ilgi jābūt katrā punktā), lai var ātri visam piekļūt
    total_time = {}
    for i in range(vrp['stops']):
        total_time[i] = {}
        for j in range(vrp['stops']):
            if i == j:
                total_time[i][j] = 0
            else:
                total_time[i][j] = int(
                    service_time(vrp, i) + travel_time(vrp, i, j))

    def time_evaluator(manager, i, j):
        return total_time[manager.IndexToNode(i)][manager.IndexToNode(j)]

    return time_evaluator


# laika logu novērtētājs, pievieno constraint par tiem
def add_time_window_constraints(routing, manager, vrp, time_evaluator_index):
    time = 'Time'

    routing.AddDimension(
        time_evaluator_index,
        60,  # slack laiks – cik ilgi var gaidīt kādā punktā
        1440,  # cik ilgi kopā viena mašīna var braukt
        False,  # laiki nav jāsāk no nulles, braucienu var sākt jebkurā dienas laikā
        time)

    time_dimension = routing.GetDimensionOrDie(time)

    # pievieno laika logu constraints
    for location, time_window in enumerate(vrp['time_windows']):
        if location == 0:  # depo nav ierobežojumu
            continue
        index = manager.NodeToIndex(location)
        time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])
        routing.AddToAssignment(time_dimension.SlackVar(index))

    # katrai mašīnai pieliek laika loga nosacījumus
    for truck in range(vrp['trucks']):
        index = routing.Start(truck)
        time_dimension.CumulVar(index).SetRange(vrp['time_windows'][0][0], vrp['time_windows'][0][1])
        routing.AddToAssignment(time_dimension.SlackVar(index))


# lasāmi izprintē risinājumu
def print_solution(manager, routing, assignment):
    time_dimension = routing.GetDimensionOrDie('Time')
    capacity_dimension = routing.GetDimensionOrDie('Capacity')

    total_distance = 0
    total_trash = 0
    min_time = 1440
    max_time = 0

    for truck in range(manager.GetNumberOfVehicles()):
        index = routing.Start(truck)
        truck_route = 'Route for truck {}: \n'.format(truck)
        distance = 0
        start_time = 0

        while not routing.IsEnd(index):
            load_var = capacity_dimension.CumulVar(index)
            time_var = time_dimension.CumulVar(index)

            if manager.IndexToNode(index) == 0:
                start_time = assignment.Min(time_var)

                if start_time != 0:
                    min_time = min(min_time, start_time)
                truck_route += '    Stop nr. {}: time arriving – {} -> \n'.format(manager.IndexToNode(index),
                                                                                  assignment.Min(time_var))

            else:
                truck_route += '    Stop nr. {}: trash volume when arriving – {}, time arriving – {} -> \n'.format(
                    manager.IndexToNode(index),
                    assignment.Value(load_var),
                    assignment.Min(time_var))

            previous_index = index
            index = assignment.Value(routing.NextVar(index))
            distance += routing.GetArcCostForVehicle(previous_index, index, truck)
        load_var = capacity_dimension.CumulVar(index)
        time_var = time_dimension.CumulVar(index)

        # pēdējais stop
        truck_route += '    Stop nr. {}: trash volume when arriving – {},  time arriving – {} \n'.format(
            manager.IndexToNode(index),
            assignment.Value(load_var),
            assignment.Min(time_var))
        max_time = max(max_time, assignment.Min(time_var))

        truck_route += 'Route distance: {} m \n'.format(distance)
        truck_route += 'Trash volume in this route: {} m^3 \n'.format(assignment.Value(load_var))
        truck_route += 'Route time: {} min \n'.format(assignment.Value(time_var) - start_time)

        print(truck_route)

        total_distance += distance
        total_trash += assignment.Value(load_var)

    print('Total distance: {} m'.format(total_distance))
    print('Total trash volume: {} m^3'.format(total_trash))
    print('Total time needed: {} min'.format(max_time - min_time))


def draw_tours(vrp, routing=None, solution=None):
    for i in range(vrp['stops']):
        vertex = vrp['coords'][i]
        if i == 0:
            plt.scatter(x=vertex[0], y=vertex[1], c='black', zorder=2, marker='*', s=150)
        else:
            plt.scatter(x=vertex[0], y=vertex[1], c='black', zorder=2)

    for i in range(1, vrp['stops']):
        stop = vrp['coords'][i]
        plt.text(stop[0] - 500, stop[1] + 200, vrp['time_windows'][i], fontsize='xx-small')
        plt.text(stop[0] - 250, stop[1] - 100, vrp['trash'][i], fontsize='xx-small')


    if solution is not None and routing is not None:

        colormap = plt.cm.gist_ncar
        colors = [colormap(i) for i in np.linspace(0.25, 0.75, vrp['trucks'])]

        i = 0
        for truck in range(manager.GetNumberOfVehicles()):
            index = routing.Start(truck)

            prev_stop = vrp['coords'][0]
            while not routing.IsEnd(index):
                stop = vrp['coords'][manager.IndexToNode(index)]
                plt.plot([prev_stop[0], stop[0]], [prev_stop[1], stop[1]], c=colors[i], linewidth=3, zorder=1)
                prev_stop = stop
                index = solution.Value(routing.NextVar(index))
            stop = vrp['coords'][0]
            plt.plot([prev_stop[0], stop[0]], [prev_stop[1], stop[1]], c=colors[i], linewidth=3, zorder=1)
            i += 1

        for truck in range(manager.GetNumberOfVehicles()):
            index = routing.Start(truck)
            while not routing.IsEnd(index):
                i = manager.IndexToNode(index)
                if i != 0:
                    stop = vrp['coords'][i]
                    time = solution.Min(routing.GetDimensionOrDie('Time').CumulVar(index))
                    plt.text(stop[0] - 100, stop[1] - 400, time, fontsize='xx-small', color='green')
                index = solution.Value(routing.NextVar(index))

    plt.show()


if __name__ == '__main__':
    vrp = get_vrp()

    manager = pywrapcp.RoutingIndexManager(vrp['stops'], vrp['trucks'], vrp['depot'])
    routing = pywrapcp.RoutingModel(manager)

    distance_evaluator_index = routing.RegisterTransitCallback(partial(create_distance_evaluator(vrp), manager))
    routing.SetArcCostEvaluatorOfAllVehicles(distance_evaluator_index)

    demand_evaluator_index = routing.RegisterUnaryTransitCallback(partial(create_demand_evaluator(vrp), manager))
    add_capacity_constraints(routing, vrp, demand_evaluator_index)

    time_evaluator_index = routing.RegisterTransitCallback(partial(create_time_evaluator(vrp), manager))
    add_time_window_constraints(routing, manager, vrp, time_evaluator_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.FromSeconds(2)
    search_parameters.log_search = True

    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        print_solution(manager, routing, solution)
        draw_tours(vrp, routing, solution)
    else:
        draw_tours(vrp)
        print('No solution found!')
