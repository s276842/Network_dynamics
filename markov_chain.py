import numpy as np
import networkx as nx
import pandas as pd
from pprint import pprint

# create a function that maps transition probability dataframe
# to markov edges and weights

class PoissonClock():
    def __init__(self, rate):
        self.rate = rate
        self.time = 0

    def __next__(self):
        # self.time +=- np.log(np.random.rand())/self.rate
        return - np.log(np.random.rand())/self.rate

class DiscreteClock():
    def __init__(self):
        self.time = 0

    def __next__(self):
        # self.time += 1
        return 1

class MarkovChain():
    def __init__(self, transition_matrix, continuous_time=False):
        self.graph = nx.from_pandas_adjacency(transition_matrix.T, create_using=nx.DiGraph)
        self.transition_matrix = transition_matrix
        self.states = list(self.graph.nodes)

        self.w = np.sum(transition_matrix, axis=1)
        self.w_star = np.max(self.w)
        self.transition_prob = self.transition_matrix / self.w_star
        self.transition_prob += np.diag(np.ones(len(self.w)) - np.sum(self.transition_prob, axis=1))
        self.transition_prob_cum = np.cumsum(self.transition_prob, axis=1)

        self.current_state = np.random.choice(self.graph.nodes)

        if continuous_time:
            self.clock = PoissonClock(self.w_star)
        else:
            self.clock = DiscreteClock()


    def __call__(self, state):
        self.current_state = state

    def __iter__(self):
        while True:
            next_state, next_tick = self.__next__()
            self.current_state = next_state
            yield (next_state, next_tick)


    def __next__(self):
        next_state =np.random.choice(
            self.states,
            p = self.transition_prob.loc[self.current_state]
        )
        # next_state = self.states[np.argwhere(self.transition_prob_cum.loc[self.current_state].values > np.random.rand())[0][0]]
        next_tick = next(self.clock)
        return (next_state, next_tick)

    def walk(self, init_state = None, num_steps=1000, till_first_return = False, hitting_set = []):
        future_states = []
        times = []
        if init_state is not None:
            self.current_state = init_state

        future_states.append(self.current_state)
        times.append(0)

        for i, x in enumerate(self.__iter__()):
            next_state, next_time = x
            # if next_state == future_states[i]:
            #     times[i] += next_time
            #     continue
            # else:
            future_states.append(next_state)
            times.append(next_time)

            if self.__is_walk_end(i,num_steps,till_first_return,init_state, hitting_set):
                return (np.array(future_states), np.cumsum(times))



    def __is_walk_end(self, i, num_steps, till_first_return, init_state, hitting_set):
        if till_first_return:
            if self.current_state == init_state:
                return True
        elif i >= num_steps-1:
            return True
        elif self.current_state in hitting_set:
            return True
        else:
            return False

    def particles(self, init_state, num_particles = 100, till_first_return=False):
        walks = [self.walk(init_state, till_first_return=till_first_return) for x in range(num_particles)]
        # print(np.mean([x[1][-1] for x in walks]))
        print([x[1][-1] for x in walks])
        return walks


if __name__ == '__main__':


    # transition_prob = {'Sunny': {'Sunny': {'weight':0.8}, 'Rainy':{'weight':0.19}, 'Snowy':{'weight':0.01}}, 'Rainy': {'Sunny': {'weight':0.2}, 'Rainy':{'weight':0.7}, 'Snowy':{'weight':0.1}}, 'Snowy': {'Sunny':{'weight': 0.1}, 'Rainy':{'weight':0.2}, 'Snowy':{'weight':0.7}}}
    # weather_chain = MarkovChain(transition_prob)
    #
    #
    # chain = weather_chain.generate_states(current_state='Sunny', no=100)
    # print(np.unique(chain, return_counts=True))

    # Problem 1
    nodes = ['o', 'a', 'b', 'c', 'd']
    transition_matrix = pd.DataFrame(np.zeros((len(nodes),len(nodes))), columns=nodes, index=nodes)
    transition_matrix.loc['o'] = [0, 2/5, 1/5, 0, 0]
    transition_matrix.loc['a'] = [0, 0, 3/4, 1/4, 0]
    transition_matrix.loc['b'] = [1/2, 0, 0, 1/2, 0]
    transition_matrix.loc['c'] = [0, 0, 1/3, 0, 2/3]
    transition_matrix.loc['d'] = [0, 1/3, 0, 1/3, 0]


    chain = MarkovChain(transition_matrix, continuous_time=True)
    print(chain.walk(init_state='o', till_first_return=False, num_steps=1))

    # print(np.mean([chain.walk(init_state='a', till_first_return=True)[1][-1] for x in range(10000)]))


    # print(np.mean([np.mean([chain.walk(init_state='o', hitting_set='d')[1][-1] for x in range(1000)]) for y in range(10)]))
    #
    # Q = chain.transition_prob
    # values, vectors = np.linalg.eig(Q.T)
    # index = np.argmax(values.real)
    # pi_bar = vectors[:, index].real
    # pi_bar = pi_bar / np.sum(pi_bar)
    # print("pi_bar=", pi_bar)
    #
    # print(f"Return time 'a' Ea[Ta] = 1/(w*pi_bar_a) = {1/(chain.w['a']*pi_bar[1]):.2f}")
    #
    # # P = np.diag(1 / (transition_matrix @ np.ones(len(transition_matrix)))) @ transition_matrix
    # P = np.diag(1/chain.w)@chain.transition_matrix
    # P.index = nodes
    # S = ['a']
    # # same as R=np.setdiff1d(np.array(range(n)),S)
    # R = [node for node in nodes if node not in S]
    #
    # # Restrict P to R x R to obtain hat(P)
    # hatP = P.loc[R,R]
    # hatx = np.linalg.solve((np.ones(len(hatP)) - hatP), np.array(1 / chain.w[R]))

    walks = chain.particles(init_state='a', till_first_return=True)