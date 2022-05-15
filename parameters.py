# Source:
# Strombom D, Mann RP,
# Wilson AM, Hailes S, Morton AJ, Sumpter DJT,
# King AJ. 2014 Solving the shepherding
# problem: heuristics for herding autonomous,
# interacting agents. J. R. Soc. Interface 11:
# 20140719.
# http://dx.doi.org/10.1098/rsif.2014.071

field_length = 120      # field width and height
num_agents = 1
num_nearest = num_agents-1       # number of nearest neighbors for LCM
r_s = 30        # shepherd detection distance
r_a = 2     # agent repulsion radius
agent_init = 30     # initial agent field length
target_radius = 20     # distance from target to trigger win condition
delta_s = 1.5       # shepherd displacement per time step
delta_a = 1     # agent displacement per time step
# p_a > c > p_s > h, see source citation for parameter details
p_a = 2
c = 1.05
p_s = 1
h = .5
# remove random agent movements for now
e = 0   # = .3
graze_prob = 0  # = 0.05
frame_reset = 600   # automatically reset game after x frames