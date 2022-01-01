import numpy as np

alpha = 0.1
beta = 1
gamma = 1
nr_trials = 50

class World():
    def __init__(self, world_data):
        self.dim = world_data['dimensions']
        self.walls = world_data['walls']
        self.slippery_states = world_data['slippery_states']
        self.world = self.build_world()
        self.visited = np.zeros(self.dim)
        self.s_start = world_data['s_start']
        self.s_terminal = world_data['s_terminal']
        self.rewards = world_data['rewards']
        self.epsilon = 0.1

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

class Agent(World):
    def __init__(self, world_data, id, method, selection):
        super().__init__(world_data)
        self.id = id
        self.method = method
        self.selection = selection
        self.location = self.s_start
        self.prev_location = self.s_start
        self.actions = np.array([0,1,2,3]) #0:up,1:down,2:left,3:right
        self.Q = np.zeros((len(self.actions), self.dim[0], self.dim[1]))

    def eGreedy(self,actions,qvals):
        if(np.random.uniform(0,1) < self.epsilon):
            return np.random.choice(actions)
        else:
            best_actions = np.where(qvals == np.max(qvals))
            nr = np.random.choice(best_actions[0])
        return actions[nr]

    def softmax(self,actions,qvals):
        f_qvals = np.exp(qvals - np.max(qvals)) / np.sum(np.exp(qvals - np.max(qvals)))
        nr = np.random.choice(np.where(f_qvals == np.max(f_qvals))[0])
        return actions[nr]

    def run(self):
        if self.method == "Q-learning":
            nr_moves = self.Qlearning()
        if self.method == "Sarsa":
            nr_moves = self.Sarsa()
        return nr_moves

    def get_actions(self):
        possible_actions = []
        qvals  = []
        location = self.get_location(self.location)
        for action in self.actions:
            self.move(action)
            if self.location != -1:
                possible_actions.append(action)
                qvals.append(self.Q[action, location[0], location[1]])
            self.undo_move(action)
        return possible_actions, qvals

    def Qlearning(self):
        moves = []
        for trial in range(nr_trials):
            move = 0
            stop = False
            while self.location != self.s_terminal and stop != True:
                #chose action
                actions,qvals = self.get_actions()
                if self.selection == "eGreedy":
                    action = self.eGreedy(actions,qvals)
                else:
                    action = self.softmax(actions,qvals)
                loc0 = self.get_location(self.location)
                self.move(action)
                self.slippery()
                if self.get_reward == -100:
                    stop = True
                loc1 = self.get_location(self.location)
                #update Q
                reward = self.get_reward()
                self.Q[action, loc0[0], loc0[1]] = self.Q[action,loc0[0],loc0[1]] + alpha * (reward + gamma * np.max(self.Q[:,loc1[0],loc1[1]]) - self.Q[action,loc0[0],loc0[1]])
                if trial == 49:
                    self.visited[loc1[0], loc1[1]] += 1
                move += 1
            moves.append(move)
            self.location = self.s_start
        return moves

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
                if trial == 49:
                    self.visited[loc1[0], loc1[1]] += 1
                move += 1
            moves.append(move)
            self.location = self.s_start
        return moves

    def get_location(self, position):
        return position//self.dim[1], position%self.dim[1]

    def set_location(self, position):
        if position[0] < 0 or position[0] >= self.dim[0]:
            return -1 #location does not exist
        if position[1] < 0 or position[1] >= self.dim[1]:
            return -1
        return int(self.world[position[0]][position[1]])

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

    def undo_move(self, direction):
        self.location = self.prev_location

    def get_reward(self):
        return self.rewards[self.location]

    def slippery(self):
        if self.location in self.slippery_states:
            nr = np.random.uniform(0,1)
            if nr < 0.4:
                self.move(1)
