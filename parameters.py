# Source:
# Strombom D, Mann RP,
# Wilson AM, Hailes S, Morton AJ, Sumpter DJT,
# King AJ. 2014 Solving the shepherding
# problem: heuristics for herding autonomous,
# interacting agents. J. R. Soc. Interface 11:
# 20140719.
# http://dx.doi.org/10.1098/rsif.2014.071

num_agents = 50
num_nearest = 10       # number of nearest neighbors for LCM
r_s = 35        # shepherd detection distance
r_a = 2     # agent repulsion radius
init_field = 20     # initial agent field length
delta_s = 1.5       # shepherd displacement per time step
delta_a = 1     # agent displacement per time step
# p_a > c > p_s > h, see source citation for parameter details
p_a = 2
c = 1.05
p_s = 1
h = .5
# e = .3
# graze_prob = 0.05
# remove random agent movements for now
e = 0
graze_prob = 0