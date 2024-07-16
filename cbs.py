import copy
import time as timer
import heapq
import random
from single_agent_planner import compute_heuristics, a_star, get_location, get_sum_of_cost


def detect_collision(path1, path2):
    ##############################
    # Task 3.1: Return the first collision that occurs between two robot paths (or None if there is no collision)
    #           There are two types of collisions: vertex collision and edge collision.
    #           A vertex collision occurs if both robots occupy the same location at the same timestep
    #           An edge collision occurs if the robots swap their location at the same timestep.
    #           You should use "get_location(path, t)" to get the location of a robot at time t.

    # vertex
    max_len = max(len(path1), len(path2))
    for ts in range(max_len):
        loc_p1 = get_location(path1, ts)
        loc_p2 = get_location(path2, ts)

        if loc_p2 == loc_p1:
            return [loc_p1], ts
        if ts < max_len - 1:
            next_ts = ts + 1
            loc_p1_next = get_location(path1, next_ts)
            loc_p2_next = get_location(path2, next_ts)
            if loc_p1 == loc_p2_next and loc_p2 == loc_p1_next:
                return [loc_p1, loc_p2_next], next_ts
    return None


def detect_collisions(paths):
    ##############################
    # Task 3.1: Return a list of first collisions between all robot pairs.
    #           A collision can be represented as dictionary that contains the id of the two robots, the vertex or edge
    #           causing the collision, and the timestep at which the collision occurred.
    #           You should use your detect_collision function to find a collision between two robots.

    result = []
    for agent_index in range(len(paths)):
        for other_agent_index in range(agent_index + 1, len(paths)):
            temp = detect_collision(paths[agent_index], paths[other_agent_index])
            # print(temp)
            if temp is not None:
                temp_coll = {'agent1': agent_index, 'agent2': other_agent_index, 'loc': temp[0],
                             'timestep': temp[1]}
                result.append(temp_coll)
    # print(result)
    return result


def standard_splitting(collision):
    ##############################
    # Task 3.2: Return a list of (two) constraints to resolve the given collision
    #           Vertex collision: the first constraint prevents the first agent to be at the specified location at the
    #                            specified timestep, and the second constraint prevents the second agent to be at the
    #                            specified location at the specified timestep.
    #           Edge collision: the first constraint prevents the first agent to traverse the specified edge at the
    #                          specified timestep, and the second constraint prevents the second agent to traverse the
    #                          specified edge at the specified timestep

    result = []
    if len(collision['loc']) == 1:  # vertex collision
        temp = {'agent': collision['agent1'], 'loc': collision['loc'], 'timestep': collision['timestep'], 'pos': False}
        temp1 = {'agent': collision['agent2'], 'loc': collision['loc'], 'timestep': collision['timestep'], 'pos': False}
        result.append(temp)
        result.append(temp1)
    else:
        temp = {'agent': collision['agent1'], 'loc': collision['loc'], 'timestep': collision['timestep'], 'pos': False}
        temp1 = {'agent': collision['agent2'], 'loc': [collision['loc'][1], collision['loc'][0]], 'timestep': collision['timestep'],
                 'pos': False}
        result.append(temp)
        result.append(temp1)
    #print(result)
    return result


def disjoint_splitting(collision):
    ##############################
    # Task 4.1: Return a list of (two) constraints to resolve the given collision
    #           Vertex collision: the first constraint enforces one agent to be at the specified location at the
    #                            specified timestep, and the second constraint prevents the same agent to be at the
    #                            same location at the timestep.
    #           Edge collision: the first constraint enforces one agent to traverse the specified edge at the
    #                          specified timestep, and the second constraint prevents the same agent to traverse the
    #                          specified edge at the specified timestep
    #           Choose the agent randomly

    location = collision['loc']
    timestep = collision['timestep']
    constraints = []
    agent1 = collision['agent1']
    agent2 = collision['agent2']
    if random.randint(0, 1) == 0:
        if len(location) == 1:
            constraints.append({'agent': agent1, 'loc': location, 'timestep': timestep, 'pos': True})
            constraints.append({'agent': agent1, 'loc': location, 'timestep': timestep, 'pos': False})
        else:
            constraints.append(
                {'agent': agent1, 'loc': location, 'timestep': timestep, 'pos': True})
            constraints.append(
                {'agent': agent1, 'loc': location, 'timestep': timestep, 'pos': False})
    else:
        if len(location) == 1:
            constraints.append({'agent': agent2, 'loc': location, 'timestep': timestep, 'pos': True})
            constraints.append({'agent': agent2, 'loc': location, 'timestep': timestep, 'pos': False})
        else:
            constraints.append(
                {'agent': agent2, 'loc': location.reverse(), 'timestep': timestep, 'pos': True})
            constraints.append(
                {'agent': agent2, 'loc': location.reverse(), 'timestep': timestep, 'pos': False})
    return constraints


def paths_violate_constraint(constraint, paths):
    assert constraint['pos'] is True
    result = []
    for path_index in range(len(paths)):
        if path_index == constraint['agent']:
            continue
        curr = get_location(paths[path_index], constraint['timestep'])
        prev = get_location(paths[path_index], constraint['timestep'] - 1)
        if len(constraint['loc']) == 1:  # vertex constraint
            if constraint['loc'][0] == curr:
                result.append(path_index)
        else:  # edge constraint
            if constraint['loc'][0] == prev or constraint['loc'][1] == curr \
                    or constraint['loc'] == [curr, prev]:
                result.append(path_index)
    return result


class CBSSolver(object):
    """The high-level search of CBS."""

    def __init__(self, my_map, starts, goals):
        """my_map   - list of lists specifying obstacle positions
        starts      - [(x1, y1), (x2, y2), ...] list of start locations
        goals       - [(x1, y1), (x2, y2), ...] list of goal locations
        """

        self.my_map = my_map
        self.starts = starts
        self.goals = goals
        self.num_of_agents = len(goals)

        self.num_of_generated = 0
        self.num_of_expanded = 0
        self.CPU_time = 0

        self.open_list = []

        # compute heuristics for the low-level search
        self.heuristics = []
        for goal in self.goals:
            self.heuristics.append(compute_heuristics(my_map, goal))

    def push_node(self, node):
        heapq.heappush(self.open_list, (node['cost'], len(node['collisions']), self.num_of_generated, node))
        print("Generate node {}".format(self.num_of_generated))
        self.num_of_generated += 1

    def pop_node(self):
        _, _, id, node = heapq.heappop(self.open_list)
        print("Expand node {}".format(id))
        self.num_of_expanded += 1
        return node

    def find_solution(self, disjoint=True):
        """ Finds paths for all agents from their start locations to their goal locations

        disjoint    - use disjoint splitting or not
        """

        self.start_time = timer.time()

        # Generate the root node
        # constraints   - list of constraints
        # paths         - list of paths, one for each agent
        #               [[(x11, y11), (x12, y12), ...], [(x21, y21), (x22, y22), ...], ...]
        # collisions     - list of collisions in paths
        root = {'cost': 0,
                'constraints': [],
                'paths': [],
                'collisions': []}
        for i in range(self.num_of_agents):  # Find initial path for each agent
            path = a_star(self.my_map, self.starts[i], self.goals[i], self.heuristics[i],
                          i, root['constraints'])
            if path is None:
                raise BaseException('No solutions')
            root['paths'].append(path)

        root['cost'] = get_sum_of_cost(root['paths'])
        root['collisions'] = detect_collisions(root['paths'])
        self.push_node(root)

        # Task 3.1: Testing
        print(root['collisions'])

        # Task 3.2: Testing
        for collision in root['collisions']:
            print(standard_splitting(collision))

        ##############################
        # Task 3.3: High-Level Search
        #           Repeat the following as long as the open list is not empty:
        #             1. Get the next node from the open list (you can use self.pop_node()
        #             2. If this node has no collision, return solution
        #             3. Otherwise, choose the first collision and convert to a list of constraints (using your
        #                standard_splitting function). Add a new child node to your open list for each constraint
        #           Ensure to create a copy of any objects that your child nodes might inherit

        while len(self.open_list) != 0:
            p = self.pop_node()
            if not p['collisions']:
                return p['paths']

            colli = p['collisions'][0]
            if disjoint:
                cons = disjoint_splitting(colli)
            else:
                cons = standard_splitting(colli)
            for con in cons:
                q = dict(cost=0, constraints=[], paths=[], collisions=[])
                q['constraints'] = p['constraints'] + [con]
                q['paths'] = copy.deepcopy(p['paths'])
                ai = con['agent']
                path = a_star(self.my_map, self.starts[ai], self.goals[ai], self.heuristics[ai], ai, q['constraints'])
                abandon = False
                if path is not None:
                    if con['pos']:
                        ids = paths_violate_constraint(con, p['paths'])
                        for other_ai in ids:
                            other_path = a_star(self.my_map, self.starts[other_ai], self.goals[other_ai],
                                                self.heuristics[other_ai], other_ai, q['constraints'])
                            if other_path is not None:
                                q['paths'][other_ai] = other_path
                            else:
                                abandon = True
                                break
                    q['paths'][ai] = path
                    q['collisions'] = detect_collisions(q['paths'])
                    q['cost'] = get_sum_of_cost(q['paths'])
                    if not abandon:
                        self.push_node(q)

        self.print_results(root)
        return root['paths']

    def print_results(self, node):
        print("\n Found a solution! \n")
        CPU_time = timer.time() - self.start_time
        print("CPU time (s):    {:.2f}".format(CPU_time))
        print("Sum of costs:    {}".format(get_sum_of_cost(node['paths'])))
        print("Expanded nodes:  {}".format(self.num_of_expanded))
        print("Generated nodes: {}".format(self.num_of_generated))
