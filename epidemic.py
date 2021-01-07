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
        val = np.random.choice(nodes_ind, int(c), p=nodes_prob, replace=False) #replace=False to avoid multiple link between the same nodes

        edges_to_add = [(i, ind) for ind in val]
        G.add_node(i)
        G.add_edges_from(edges_to_add)

    return G



def simulate_epidemic(G, beta, rho, time=15, vaccination=None, num_infected=10):
    state_dict = {state:set() for state in ['S', 'I', 'R', 'V']}

    # Initialize nodes states
    nodes = G.nodes()
    infected = np.random.choice(nodes, num_infected)
    state_dict['I'].update(infected)
    state_dict['S'].update([node for node in nodes if node not in infected])

    if vaccination is not None:
        try:

            percentage_vaccination = vaccination[0]
            num_vaccination = int(percentage_vaccination * len(nodes) / 100)
            not_vaccinated = [(node, state) for state in ['S', 'I', 'R'] for node in state_dict[state]]

            # vaccinated = np.random.choice(not_vaccinated, num_vaccination)
            vaccinated = np.random.choice(np.arange(len(not_vaccinated)), num_vaccination, replace=False)
            for ind in vaccinated:
                node, state = not_vaccinated[ind]
                state_dict[state].remove(node)
                state_dict['V'].add(node)
        except:
            pass

    # Log initial state
    states = pd.DataFrame(map(len, state_dict.values()), index=state_dict.keys(), columns=[0])

    for t in range(1, time + 1):
        try:
            if vaccination is not None:
                percentage_vaccination = vaccination[t] - vaccination[t-1]
                num_vaccination = int(percentage_vaccination * len(nodes) / 100)
                not_vaccinated = [(node, state) for state in ['S', 'I', 'R'] for node in state_dict[state]]

                # vaccinated = np.random.choice(not_vaccinated, num_vaccination)
                vaccinated = np.random.choice(np.arange(len(not_vaccinated)), num_vaccination, replace=False)
                for ind in vaccinated:
                    node, state = not_vaccinated[ind]
                    state_dict[state].remove(node)
                    state_dict['V'].add(node)
        except:
            pass

        # Extract infected and susceptibles
        infected = state_dict['I']
        susceptibles = state_dict['S']

        new_infected = []
        for node in susceptibles:
            num_infected_neighbors = sum([1 for neigh in G.neighbors(node) if neigh in infected])
            if np.random.rand() < 1 - (1 - beta) ** num_infected_neighbors:
                new_infected.append(node)

        # Compute randomly new recovered and new infected
        new_recovered = [node for node in infected if np.random.rand() < rho]

        # Update state_dict moving I -> R and S -> I
        for node in new_infected:
            state_dict['S'].remove(node)
            state_dict['I'].add(node)

        for node in new_recovered:
            state_dict['I'].remove(node)
            state_dict['R'].add(node)

        # Log states
        states[t] = [len(nodes) for nodes in state_dict.values()]

    if vaccination is None:
        states.drop('V', axis=0, inplace=True)
    return states

    # def simulate_epidemic(self, beta, rho, time=15, vaccination=None):
    #     state_dict = {i: 'S' for i in self.graph.nodes()}
    #     infected = np.random.choice(self.graph.nodes(), 10)
    #     self.update_state_dict(state_dict, infected, 'I')
    #
    #     states = pd.DataFrame([self.graph.number_of_nodes() - 10, 10, 0], index=['S', 'I', 'R'], columns=[0])
    #
    #     if vaccination is not None:
    #         num_vaccination = int(vaccination[0] * self.graph.number_of_nodes / 100)
    #         vaccinated = np.random.choice(self.graph.nodes(), 10)
    #         states.loc['V'] = self.vaccinate(state_dict, vaccination[0])
    #
    #     for t in range(1, time + 1):
    #         # Extract infected and susceptibles
    #         infected = [node for node, state in state_dict.items() if state == 'I']
    #         susceptibles = [node for node, state in state_dict.items() if state == 'S']
    #
    #         # Count infected neighbors for each susceptible node
    #         suceptibles_inf_neighbors = {node: sum([1 for neigh in self.graph.neighbors(node) if state_dict[neigh] == 'I'])
    #                                      for node in susceptibles}
    #
    #         # Compute randomly new recovered and new infected
    #         new_recovered = [node for node in infected if np.random.rand() < rho]
    #         num_recovered = len(new_recovered)
    #         new_infected = [node for node in susceptibles if
    #                         np.random.rand() < 1 - (1 - beta) ** suceptibles_inf_neighbors[node]]
    #         num_infected = len(new_infected)
    #
    #         # Update state_dict moving I -> R and S -> I
    #         for node in new_recovered:
    #             state_dict[node] = 'R'
    #         for node in new_infected:
    #             state_dict[node] = 'I'
    #
    #         # Log current distribution of states
    #         states[t] = [states.loc['S', t - 1] - num_infected,
    #                      states.loc['I', t - 1] + num_infected - num_recovered,
    #                      states.loc['R', t - 1] + num_recovered]
    #
    #     return states


def plot_pandemic(mean, std, states=['S', 'I', 'R']):
    fig, ax = plt.subplots(1)
    colors = ['blue', 'red', 'green', 'purple']
    for i, (state, color) in enumerate(zip(states, colors)):
        ax.plot(np.arange(16), mean[i], label=state)
        ax.fill_between(np.arange(16), mean[i] + std[i],
                        mean[i] - std[i], alpha=0.2)
    ax.legend(loc='upper left')
    plt.show()


def repeat_simulation(G, beta, rho, time, vaccination, n_repetitions=100, num_infected=10):
    simulations = []
    for n in range(n_repetitions):
        df_sim = simulate_epidemic(G, beta=beta, rho=rho, time=time, vaccination=vaccination, num_infected=num_infected)
        simulations.append(df_sim.values)
    simulations = np.array(simulations)
    return (simulations.mean(axis=0), simulations.std(axis=0))

if __name__ == '__main__':

    # 1.1 - Epidemic simulation
    G = regular_symmetric_graph(nodes=500, k=4)
    mean_distr, std_distr = repeat_simulation(G, beta=0.3, rho=0.7, time=15, vaccination=None, n_repetitions=100)

    plot_pandemic(mean_distr, std_distr)
    plt.plot(np.arange(16), -np.diff([500] + [x for x in mean_distr[0]]))
    plt.show()



    # 1.2 - Generating random graph
    G = random_graph(nodes=900, avg_degree=3.14)
    degrees = [degree for node, degree in G.degree]
    print(f"Node {np.argmax(degrees)} has the maximum degree with value {np.max(degrees)}")
    print(f"Node {np.argmin(degrees)} has the maximum degree with value {np.min(degrees)}")
    print(f"The average degree of the graph is {np.mean(degrees)}")

    # 2 - Pandemic without vaccination
    G = random_graph(nodes=500, avg_degree=6)
    mean_distr, std_distr = repeat_simulation(G, beta=0.3, rho=0.7, time=15, vaccination=None, n_repetitions=100)

    plot_pandemic(mean_distr, std_distr)
    plt.plot(np.arange(16), -np.diff([500] + [x for x in mean_distr[0]]))
    plt.show()


    # 3 - Pandemic with vaccination
    G = random_graph(nodes=500, avg_degree=6)
    vaccination_percentages = [0, 5, 15, 25, 35, 45, 55, 60]
    mean_distr, std_distr = repeat_simulation(G, beta=0.3, rho=0.7, time=15, vaccination=vaccination_percentages, n_repetitions=100)

    plot_pandemic(mean_distr, std_distr, states=['S', 'I', 'R', 'V'])
    plt.plot(np.arange(16), -np.diff([500] + [x for x in mean_distr[0]]))
    plt.show()


    # 4 - H1N1 simulation

    # from week 42 of 2009 to week 5 of 2010
    weeks = np.hstack((np.arange(42, 54), np.arange(1, 6)))
    real_infected_distr = np.array([1, 1, 3, 5, 9, 17, 32, 32, 17, 5, 2, 1, 0, 0, 0, 0])
    vaccination_percentages = [5, 9, 16, 24, 32, 40, 47, 54, 59, 60]

    import itertools
    k_zero = 10
    beta_zero = 0.3
    rho_zero = 0.6
    delta_k = 1
    delta_beta = 0.1
    delta_rho = 0.1
    n_rep = 10
    min_rmse = np.inf
    best_k = k_zero
    best_beta = beta_zero
    best_rho = rho_zero
    while True:
        k_values = [best_k - delta_k, best_k, best_k + delta_k]
        beta_values = [best_beta - delta_beta, best_beta, best_beta + delta_beta]
        rho_values = [best_rho - delta_rho, best_rho, best_rho + delta_rho]

        for k, beta, rho in itertools.product(k_values, beta_values, rho_values):
            G = random_graph(934, avg_degree=k)
            mean_distr, std_distr = repeat_simulation(G, beta=beta, rho=rho, vaccination=vaccination_percentages, time=15, n_repetitions=n_rep)

            new_infected = -np.diff([934] + [x for x in mean_distr[0]])
            distance = new_infected - real_infected_distr

            rmse = np.sqrt(distance.dot(distance) / 15)
            if rmse < min_rmse:
                print(rmse)
                min_rmse = rmse
                best_k = k
                best_beta = beta
                best_rho = rho



