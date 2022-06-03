# Source:
# Strombom D, Mann RP,
# Wilson AM, Hailes S, Morton AJ, Sumpter DJT,
# King AJ. 2014 Solving the shepherding
# problem: heuristics for herding autonomous,
# interacting agents. J. R. Soc. Interface 11:
# 20140719.
# http://dx.doi.org/10.1098/rsif.2014.071

FIELD_LENGTH = 100      # field width and height
MAX_NUM_AGENTS = 5      # number of agents [0, MAX_NUM_AGENTS]
R_S = 25        # shepherd detection distance
R_A = 2     # agent repulsion distance
TARGET_RADIUS = 20     # distance from target to trigger win condition
# P_A > P_C > P_S (eliminate inertial term for discrete setting)
P_A = 2     # agent repulsion weight
P_C = 1.2    # LCM attraction weight
P_S = 1     # shepherd repulsion weight
FRAME_RESET = 500   # automatically reset game after x frames
SAVE_TARGET = 10    # save and update target after x games
CNN_NETWORK = False     # alternatively, use linear network