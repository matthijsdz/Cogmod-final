import numpy as np
import matplotlib.pyplot as plt
from cliff_world import Agent,World

nr_agents = 10

world1 = dict({
    "s_start" : 50,
    "s_terminal" : 59,
    "dimensions" : np.array([6,10]),
    "rewards" : np.array([0]*51 + [-100]*8 + [1000]),
    "walls" : [],
    "slippery_states" : []
})
world2 = dict({
    "s_start" : 50,
    "s_terminal" : 59,
    "dimensions" : np.array([6,10]),
    "rewards" : np.array([0]*51 + [-100]*8 + [1000]),
    "walls" : [],
    "slippery_states" : [41,42,43,44,45,46,47,48]
})
world3 = dict({
    "s_start" : 50,
    "s_terminal" : 59,
    "dimensions" : np.array([6,10]),
    "rewards" : np.array([-0.01]*30 + [0]*21 + [-100]*8 + [1000]),
    "walls" : [31,32,33,34,35,36,37,38],
    "slippery_states" : []
})

totalmoves1 = []
visited1 = np.zeros((6,10))
for i in range(nr_agents):
    A1 = Agent(world1, 1, "Sarsa", "eGreedy")
    moves = A1.run()
    totalmoves1.append(moves)
    visited1 = np.add(visited1, A1.visited)

totalmoves2 = []
visited2 = np.zeros((6,10))
for i in range(nr_agents):
    A1 = Agent(world1, 1, "Sarsa", "softmax")
    moves = A1.run()
    totalmoves2.append(moves)
    visited2 = np.add(visited2, A1.visited)

totalmoves3 = []
visited3 = np.zeros((6,10))
for i in range(nr_agents):
    A1 = Agent(world1, 1, "Q-learning", "eGreedy")
    moves = A1.run()
    totalmoves3.append(moves)
    visited3 = np.add(visited3, A1.visited)

totalmoves4 = []
visited4 = np.zeros((6,10))
for i in range(nr_agents):
    A1 = Agent(world1, 1, "Q-learning", "softmax")
    moves = A1.run()
    totalmoves4.append(moves)
    visited4 = np.add(visited4, A1.visited)

fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)

#ax1.matshow(visited1, cmap=plt.cm.Blues)
#ax2.matshow(visited2, cmap=plt.cm.Blues)
#ax3.matshow(visited3, cmap=plt.cm.Blues)
#ax4.matshow(visited4, cmap=plt.cm.Blues)
total1 = np.array([0]*50)
for moves in totalmoves1:
    total1 = np.add(total1, moves)
for i in range(len(total1)):
    total1[i] = total1[i] / 10

total2 = np.array([0]*50)
for moves in totalmoves2:
    total2 = np.add(total2, moves)
for i in range(len(total2)):
    total2[i] = total2[i] / 10

total3 = np.array([0]*50)
for moves in totalmoves3:
    total3 = np.add(total3, moves)
for i in range(len(total3)):
    total3[i] = total3[i] / 10

total4 = np.array([0]*50)
for moves in totalmoves4:
    total4 = np.add(total4, moves)
for i in range(len(total4)):
    total4[i] = total4[i] / 10

y = np.arange(0,50, 1)
ax1.plot(y, total1)
ax2.plot(y, total2)
ax3.plot(y, total3)
ax4.plot(y, total4)
ax1.title.set_text("Sarsa with eGreedy")
ax2.title.set_text("Sarsa with softmax")
ax3.title.set_text("Q-learning with eGreedy")
ax4.title.set_text("Q-learning with softmax")
plt.show()
