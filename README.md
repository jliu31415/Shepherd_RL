# Shepherding Problem - DeepQ Reinforcement Learning
## Usage
Terminal commands:
* python environment.py (run game loop with user keyboard input)
    * press 'escape' key to quit pygame window
    * press 'r' key to reset current game
* python dqn_agent.py -t -r -p
    * If no flags present, dqn_agent will load the model in evaluation mode
    * -t trains the existing model
    * -t -r trains the model from scratch

The model path, as well as other simulation parameters, are declared in parameters.py. Dependencies are specified in requirements.txt.

## Description
Shepherding problem inspired by [Strombom D, Mann RP, Wilson AM, Hailes S, Morton AJ, Sumpter DJT, King AJ. 2014 Solving the shepherding problem: heuristics for herding autonomous, interacting agents. J. R. Soc. Interface 11: 20140719.](https://royalsocietypublishing.org/doi/pdf/10.1098/rsif.2014.0719) Agents are attracted to local center of mass (LCM) of neighboring agents and directly repelled by the shepherd. There is also local repulsion between agents if they move too close to each other. The environment.py file models these interactions in a discrete square game space with length FIELD_LENGTH. On each training episode, the game initializes 1 shepherd and [1, MAX_NUM_AGENTS] agents (uniformly distributed). The game ends when the global center of mass (GCM) of all sheep are within TARGET_RADIUS units of target. The reset function in environment.py determines initial positions of agents, shepherd, and target.

## DeepQ Model
The DeepQ model uses PyTorch library. There are two dqn model files - a fully connected linear (dqn_linear) and convolutional (dqn_cnn) neural network. Refer to the get_state function in environment.py for the network input dimensions. The reward function is currently determined by the agents' GCM distance to target and the minimum distance between the agents and the shepherd. The latter constraint encourages to shepherd to approach the agents to speed up the training process.

## Training
OUTPUT and RENDER parameters toggle detailed terminal output and pygame display. Setting USER_INPUT to True lets the user seed the dqn_agent memory with manual gameplay with keyboard input. While training, the game resets every FRAME_RESET frames and saves every SAVE_TARGET episodes. The model path is specified by MODEL_PATH and points to the /model subdirectory.

## Model History
Update model history as necessary
* model_linear_5.pth: MAX_NUM_AGENTS = 5 trained with dqn_linear model.

## Authors
* James Liu | Professor Wei (2021-22)