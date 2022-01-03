import numpy as np
import matplotlib.pyplot as plt

alpha = 0.1
beta = 1
gamma = 1
nr_trials = 50

#class for cliff world
class World():
    def __init__(self, world_data):
        self.dim = world_data['dimensions'] # dimensions of the world
        self.walls = world_data['walls']
        self.slippery_states = world_data['slippery_states']
        self.world = self.build_world()
        self.visited = np.zeros(self.dim) # used to  save final path the agent learns
        self.s_start = world_data['s_start']
        self.s_terminal = world_data['s_terminal'] # terminal state
        self.rewards = world_data['rewards']
        self.epsilon = 0.1 # epsilon value for e-Greedy selectino

    #Each state gets a number; States that are walls get number -1
    #signifying that the are not reachable
    def build_world(self):
        world = np.zeros(self.dim)
        for i in range(self.dim[0]):
            for j in range(self.dim[1]):
                state =  i*self.dim[1] + j
                if state in self.walls:
                    world[i][j] = -1
                else:
                    world[i][j] = state;
        return world

#Agent class inherits from world
class Agent(World):
    def __init__(self, world_data, id, method, selection):
        super().__init__(world_data)
        self.id = id
        self.method = method # learning rule [Q-learning or Sarsa]
        self.selection = selection #Selection strategy [e-Greedy or Softmax]
        self.location = self.s_start
        self.prev_location = self.s_start #For undoing the latest move
        self.actions = np.array([0,1,2,3]) #0:up,1:down,2:left,3:right
        self.Q = np.zeros((len(self.actions), self.dim[0], self.dim[1]))

    #Choses one of the available actions based on the Q-values
    #Using e-Greedy action selection
    def eGreedy(self,actions,qvals):
        if(np.random.uniform(0,1) < self.epsilon):
            return np.random.choice(actions)
        else:
            best_actions = np.where(qvals == np.max(qvals))
            nr = np.random.choice(best_actions[0])
        return actions[nr]

    #Choses one of the available actions based on the Q-values
    #Using Softmax action selection
    def softmax(self,actions,qvals):
        f_qvals = np.exp(qvals - np.max(qvals)) / np.sum(np.exp(qvals - np.max(qvals)))
        nr = np.random.choice(actions, p=f_qvals)
        return nr

    #running the agent
    def run(self):
        if self.method == "Q-learning":
            nr_moves = self.Qlearning()
        if self.method == "Sarsa":
            nr_moves = self.Sarsa()
        return nr_moves

    #returns available action. (e.g. if the agent is at the top
    #of the cliff world going "up" is not available)
    def get_actions(self):
        possible_actions = []
        qvals  = []
        location = self.get_location(self.location)
        for action in self.actions:
            self.move(action) #try action
            if self.location != -1: # if new location is a reachable state
                possible_actions.append(action)
                qvals.append(self.Q[action, location[0], location[1]])
            self.undo_move(action)
        return possible_actions, qvals

    #Function for agent with Q-learning
    def Qlearning(self):
        moves = []
        for trial in range(nr_trials):
            move = 0
            stop = False
            while self.location != self.s_terminal and stop != True:
                #chose action
                actions,qvals = self.get_actions() #get available action and Q-values
                if self.selection == "eGreedy": #select action
                    action = self.eGreedy(actions,qvals)
                else:
                    action = self.softmax(actions,qvals)
                loc0 = self.get_location(self.location) #get initial location
                self.move(action) #move
                self.slippery() #check if state is slippery
                if self.get_reward == -100: #if agent falls in cliff
                    stop = True
                loc1 = self.get_location(self.location) #get next location
                #update Q
                reward = self.get_reward()
                #Q-learning rule
                self.Q[action, loc0[0], loc0[1]] = self.Q[action,loc0[0],loc0[1]] + alpha * (reward + gamma * np.max(self.Q[:,loc1[0],loc1[1]]) - self.Q[action,loc0[0],loc0[1]])
                if trial == 49: #save the latest route in array
                    self.visited[loc1[0], loc1[1]] += 1
                move += 1
            moves.append(move)
            self.location = self.s_start
        return moves

    #Function for agent with Sarsa
    def Sarsa(self):
        moves = []
        for trial in range(nr_trials):
            actions,qvals = self.get_actions()
            if self.selection == "eGreedy":
                action = self.eGreedy(actions,qvals)
            else:
                action = self.softmax(actions,qvals)
            move = 0
            stop = False
            while self.location != self.s_terminal and stop != True:
                loc0 = self.get_location(self.location)
                #try move
                self.move(action)
                loc1 = self.get_location(self.location)
                reward = self.get_reward()
                self.undo_move(action)
                #choose move
                self.move(action)
                self.slippery()
                if self.get_reward == -100:
                    stop = True
                actions,qvals = self.get_actions()
                nextaction = self.eGreedy(actions, qvals)
                #update Q
                self.Q[action, loc0[0], loc0[1]] = self.Q[action,loc0[0],loc0[1]] + alpha * (reward + gamma * self.Q[nextaction,loc1[0],loc1[1]] - self.Q[action,loc0[0],loc0[1]])
                action = nextaction
                if trial == 49: #save the latest route in array
                    self.visited[loc1[0], loc1[1]] += 1
                move += 1
            moves.append(move)
            self.location = self.s_start
        return moves

    #in: number of state; out: [row,column]
    def get_location(self, position):
        return position//self.dim[1], position%self.dim[1]

    #in: [row, column]; out: number of state or -1 if position outside the cliff world
    def set_location(self, position):
        if position[0] < 0 or position[0] >= self.dim[0]:
            return -1 #location does not exist
        if position[1] < 0 or position[1] >= self.dim[1]:
            return -1
        return int(self.world[position[0]][position[1]])

    #move 1 step
    def move(self, direction):
        self.prev_location = self.location
        location = self.get_location(self.location)
        if direction == 0: #up
            self.location = self.set_location([location[0]-1, location[1]])
        if direction == 1: #down
            self.location = self.set_location([location[0]+1, location[1]])
        if direction == 2: #left
            self.location = self.set_location([location[0], location[1]-1])
        if direction == 3: #right
            self.location = self.set_location([location[0], location[1]+1])

    #undo latest move.
    def undo_move(self, direction):
        self.location = self.prev_location

    def get_reward(self):
        return self.rewards[self.location]

    #if location of state is slippery move one step down
    #slippery states should always be above the cliff
    def slippery(self):
        if self.location in self.slippery_states:
            nr = np.random.uniform(0,1)
            if nr < 0.4: # 40% chance to slip
                self.move(1)

if __name__ == "__main__":
    world = dict({
        "s_start" : 50,
        "s_terminal" : 59,
        "dimensions" : np.array([6,10]),
        "rewards" : np.array([-0.01]*30 + [0]*21 + [-100]*8 + [1000]),
        "walls" : [36,37,38],
        "slippery_states" : [41,42,43]
    })
    agent = Agent(world, 0, "Q-learning", "softmax")
    nr_moves = agent.run()
    visited = (agent.visited)
    plt.matshow(visited, cmap=plt.cm.Blues)
    plt.show()
