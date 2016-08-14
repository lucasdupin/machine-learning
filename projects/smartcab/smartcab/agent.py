import random
import numpy as np
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        self.q_table = {};
        self.alpha = 0.1
        self.gamma = -0.01
        self.possible_actions = ('forward', 'left', 'right', None);
        self.total_reward = 0

    def reset(self, destination=None):
        self.planner.route_to(destination)
        self.state = {};
        self.total_reward = 0

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # Update state
        self.state = {
            # "deadline": deadline,
            "light": inputs["light"],
            "oncoming": inputs["oncoming"],
            "left": inputs["left"],
            "right": inputs["right"],
            "next_waypoint": self.next_waypoint
        }

        # Unique hash to represent this state when learning
        state_hash = hash(frozenset(self.state.items()))
        # Create dictionary to hold q values
        if not state_hash in self.q_table:
            self.q_table[state_hash] = dict((a, 0) for a in self.possible_actions)

        # Select action according to your policy
        action = random.choice(self.possible_actions)
        if (random.random() > 0.01): # Avoid getting locked in local optima
            q_values = [(action, self.q_table[state_hash][action]) for action in self.possible_actions]
            action = q_values[np.argmax(q_values, axis=0)[1]][0]

        # Execute action and get reward
        reward = self.env.act(self, action) + self.gamma
        if reward < self.gamma:
            print "penalty =["
        self.total_reward += reward

        # Learn policy based on state, action, reward
        self.q_table[state_hash][action] = (1.0 - self.alpha) * self.q_table[state_hash][action] + self.alpha * reward

        # print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    # sim = Simulator(e, update_delay=0.5, display=True)  # create simulator (uses pygame when display=True, if available)
    sim = Simulator(e, update_delay=0, display=False)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
