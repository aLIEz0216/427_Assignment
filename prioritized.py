import time as timer
from single_agent_planner import compute_heuristics, a_star, get_sum_of_cost


class PrioritizedPlanningSolver(object):
    """A planner that plans for each robot sequentially."""

    def __init__(self, my_map, starts, goals):
        """my_map   - list of lists specifying obstacle positions
        starts      - [(x1, y1), (x2, y2), ...] list of start locations
        goals       - [(x1, y1), (x2, y2), ...] list of goal locations
        """

        self.my_map = my_map
        self.starts = starts
        self.goals = goals
        self.num_of_agents = len(goals)

        self.CPU_time = 0

        # compute heuristics for the low-level search
        self.heuristics = []
        for goal in self.goals:
            self.heuristics.append(compute_heuristics(my_map, goal))

    def find_solution(self):
        """ Finds paths for all agents from their start locations to their goal locations."""

        start_time = timer.time()
        result = []
        # constraints = [{'agent': 0, 'loc': [(1, 5)], 'timestep': 4, 'pos':  False}]
        # constraints = [{'agent': 0, 'loc': [(1, 5)], 'timestep': 10, 'pos':  False}]
        # constraints.append({'agent': 1, 'loc': [(1, 2), (1, 3)], 'timestep': 1, 'pos':  False})
        # constraints.append({'agent': 0, 'loc': [(1, 5)], 'timestep': 10, 'pos':  False})

        # constraints = [{'agent': 1, 'loc': [(1, 2)], 'timestep': 2, 'pos':  False},
        #                {'agent': 1, 'loc': [(1, 3)], 'timestep': 2, 'pos':  False},
        #                {'agent': 1, 'loc': [(1, 4)], 'timestep': 2, 'pos':  False}]

        constraints = []

        map_size = 0
        for row in self.my_map:
            for col in row:
                if not col:
                    map_size += 1

        for i in range(self.num_of_agents):  # Find path for each agent
            path = a_star(self.my_map, self.starts[i], self.goals[i], self.heuristics[i],
                          i, constraints)

            total_path_len = 0

            for temp_p in result:
                total_path_len += len(temp_p)

            upper_bound = map_size + total_path_len

            if path is None:
                raise BaseException('No solutions')
            result.append(path)

            ##############################
            # Task 2: Add constraints here
            #         Useful variables:
            #            * path contains the solution path of the current (i'th) agent, e.g., [(1,1),(1,2),(1,3)]
            #            * self.num_of_agents has the number of total agents
            #            * constraints: array of constraints to consider for future A* searches
            for path_index in range(len(path)):
                for agent_index in range(i + 1, self.num_of_agents):
                    new_con_v = {'agent': agent_index, 'loc': [path[path_index]], 'timestep': path_index, 'pos': False}  # vertex

                    new_con_e = {'agent': agent_index, 'loc': [path[path_index], path[path_index - 1]],  # edge
                                 'timestep': path_index, 'pos': False}
                    constraints.append(new_con_v)
                    constraints.append(new_con_e)
            # additional
            for ai in range(i + 1, self.num_of_agents):
                for time in range(len(path), upper_bound + 1):
                    constraints.append({'agent': ai, 'loc': [path[-1]], 'timestep': time, 'pos': False})

        self.CPU_time = timer.time() - start_time

        print("\n Found a solution! \n")
        print("CPU time (s):    {:.2f}".format(self.CPU_time))
        print("Sum of costs:    {}".format(get_sum_of_cost(result)))
        print(result)
        return result
