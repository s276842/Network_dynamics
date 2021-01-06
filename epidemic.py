from itertools import product
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def regular_symmetric_graph(nodes = 100, k=4):
    G = nx.Graph()
    for node in range(nodes):
        neighboors = [x%nodes  for x in range(-int(np.floor(k/2))+node, int(np.ceil(k/2) + 1 + node),1) if x != node]
#         print(neighboors)
        edges = product(*[[node], neighboors])
#         print([x for x in edges])
        G.add_edges_from(edges)
    return G


def random_graph(nodes=100, avg_degree=4):
    G = nx.complete_graph(int(avg_degree) + 1)

    for i in range(int(avg_degree) + 1, nodes):
        c = avg_degree / 2

        nodes_ind, nodes_degrees = zip(*G.degree)
        n_nodes = len(nodes_ind)
        sum_degrees = sum(nodes_degrees)
        current_avg_degree = sum_degrees / n_nodes

        if current_avg_degree >= avg_degree:
            c = int(np.floor(c))
        else:
            c = int(np.ceil(c))

        nodes_prob = np.array(nodes_degrees) / sum(nodes_degrees)
        val = np.random.choice(nodes_ind, int(c), p=nodes_prob, replace=False)

        edges_to_add = [(i, ind) for ind in val]
        G.add_node(i)
        G.add_edges_from(edges_to_add)

    return G

class Epidemic():
    def __init__(self, G):
        self.graph = G

    def update_state_dict(self, state_dict, nodes, state):
        for node in nodes:
            state_dict[node] = state

    def vaccinate(self, state_dict, percentage):
        num_vaccination = int(percentage*self.graph.number_of_nodes/100)
        vaccinated = np.random.choice(self.graph.nodes(), 10)


    def simulate_epidemic(self, beta, rho, time=15, vaccination=None):
        state_dict = {i: 'S' for i in self.graph.nodes()}
        infected = np.random.choice(self.graph.nodes(), 10)
        self.update_state_dict(state_dict, infected, 'I')

        states = pd.DataFrame([self.graph.number_of_nodes() - 10, 10, 0], index=['S', 'I', 'R'], columns=[0])

        if vaccination is not None:
            num_vaccination = int(vaccination[0] * self.graph.number_of_nodes / 100)
            vaccinated = np.random.choice(self.graph.nodes(), 10)
            states.loc['V'] = self.vaccinate(state_dict, vaccination[0])

        for t in range(1, time + 1):
            # Extract infected and susceptibles
            infected = [node for node, state in state_dict.items() if state == 'I']
            susceptibles = [node for node, state in state_dict.items() if state == 'S']

            # Count infected neighbors for each susceptible node
            suceptibles_inf_neighbors = {node: sum([1 for neigh in G.neighbors(node) if state_dict[neigh] == 'I'])
                                         for node in susceptibles}

            # Compute randomly new recovered and new infected
            new_recovered = [node for node in infected if np.random.rand() < rho]
            num_recovered = len(new_recovered)
            new_infected = [node for node in susceptibles if
                            np.random.rand() < 1 - (1 - beta) ** suceptibles_inf_neighbors[node]]
            num_infected = len(new_infected)

            # Update state_dict moving I -> R and S -> I
            for node in new_recovered:
                state_dict[node] = 'R'
            for node in new_infected:
                state_dict[node] = 'I'

            # Log current distribution of states
            states[t] = [states.loc['S', t - 1] - num_infected,
                         states.loc['I', t - 1] + num_infected - num_recovered,
                         states.loc['R', t - 1] + num_recovered]

        return states