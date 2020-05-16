# This code is based on two implementations :
#https://github.com/aborghi/retro_contest_agent
#https://github.com/simoninithomas/Deep_reinforcement_learning_Course

#Note (look for it )
#https://confluence.jetbrains.com/display/IDEADEV/Inotify+Watches+Limit

#important to install
# https://github.com/openai/retro
# https://github.com/openai/baselines
#
# follow these steps here
# https://contest.openai.com/2018-1/details/
# clone this repo:
# git clone --recursive https://github.com/openai/retro-contest.git
# source it in project (if using pycharm):
#File --> setting -->project structure ---> add content as root -->mark on all file on the left and then click source tab ,it will goes to the right .


import numpy as np
import gym

from retro_contest.local import make
from retro import make as make_retro

# This will be useful for stacking frames
#example of pong game if we gave our agent one frame , will not see if it is going
#right or left as there is no movement in one frame ,
# thats way we have to give it many frames like four games (in pong example )
from baselines.common.atari_wrappers import FrameStack
# Library used to modify frames
import cv2

#set setUseOpenCL = False means that we will not use GPU(disable OpenCL acceleration)
cv2.ocl.setUseOpenCL(False)

#Preprocessing Class

class PreprocessFrame(gym.ObservationWrapper):
    #in preprocessing :
    #we set frame to gray
    #resize the frame to 96x96x1

    def __init__(self, env):
        gym.ObservationWrapper.__init__(self,env)
        self.width = 96
        self.height = 96
        self.observation_space = gym.spaces.Box(low=0,high=255,shape=(self.height,self.width,1),dtype=np.uint8)

    def observation(self, frame):
        #set frame to grey
        #we are setting it to grey as the RGB wouldn't have any affect on our data
        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)

        #Resize the frame to 96x96x1
        frame = cv2.resize(frame,(self.width, self.height), interpolation=cv2.INTER_AREA)
        frame = frame[:,:,None] #None here is add axis at the end ( , ,1) , here 96x96 will be (96 x 96 x 1)

        return frame

class ActionsDiscretizers(gym.ActionWrapper):
    """
    wrap a gym-retro environment and make it use discrete
    actions for sonic game.

    this class consists of an array which has 12 False values (these values refers to button in action)
    like if i pressed left action , butten index will be true
    example : Left action = [F,F,F,F,F,T,F,F,F,F,F,F]
    """

    def __init__(self,env):
        super(ActionsDiscretizers,self).__init__(env)
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        actions = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'],['DOWN', 'B'], ['B']]
        self._actions = []


        """
        What we do in this loop:
        For each action in actions
            - Create an array of 12 False (12 = nb of buttons)
            For each button in action: (for instance ['LEFT']) we need to make that left button index = True
                - Then the button index = LEFT = True
            In fact at the end we will have an array where each array is an action and each elements True of this array
            are the buttons clicked.
        """

        for action in actions:
            arr = np.array([False]*12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self,a): #pylint : diable =W0221
        return self._actions[a].copy()

class RewardScaler(gym.RewardWrapper):
    """
    Bring or scale rewards to a reasonable scale for PPO as well as A2C
    This is incredibly important and effects performance
    drastically
    """
    def reward(self, reward):
        return reward * 0.01

class AllowBacktracking(gym.Wrapper):

    """
    Use deltas in max(X) as the reward, rather than deltas
    in X. This way, agents are not discouraged too heavily
    from exploring backwards if there is no way to advance
    head-on in the level.

    i.e it allows the agent to go backward without being discourage
    (avoid our agent to be stuck on a wall during the game and try to explore backward or another path)
    """

    def __init__(self,env):
        super(AllowBacktracking,self).__init__(env)
        self._cur_x = 0
        self._max_x = 0

    def reset(self,**kwargs):
        self._cur_x = 0
        self._max_x = 0
        return self.env.reset(**kwargs)

    def step(self,action):
        obs, rew ,done,info = self.env.step(action)
        self._cur_x += rew
        rew = max (0,self._cur_x - self._max_x)
        self._max_x = max(self._max_x,self._cur_x)
        return obs,rew,done,info
def make_env(env_idx):
    """
    Create an environment with some standard wrappers.
    """
    """
    The idea is that we'll build multiple instances of the environment, 
    different environments each times (different level) to avoid overfitting 
    and helping our agent to generalize better at playing sonic
    """
    # this dictionary composed of all layers in our game
    dicts = [
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'SpringYardZone.Act3'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'SpringYardZone.Act2'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'GreenHillZone.Act3'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'GreenHillZone.Act1'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'StarLightZone.Act2'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'StarLightZone.Act1'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'MarbleZone.Act2'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'MarbleZone.Act1'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'MarbleZone.Act3'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'ScrapBrainZone.Act2'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'LabyrinthZone.Act2'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'LabyrinthZone.Act1'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'LabyrinthZone.Act3'}

    ]

    # Make the environment
    print(dicts[env_idx]['game'],dicts[env_idx]['state'],flush=True)
    # record_path = "./records/" + dicts[env_idx]['state']
    env = make(game = dicts[env_idx]['game'], state=dicts[env_idx]['state'],bk2dir="./records")#record = '/tmp')

    # Build the actions array
    env = ActionsDiscretizers(env)

    # Scale the rewards
    env = RewardScaler(env)

    # PreprocessFrame
    env = PreprocessFrame(env)

    # Stack 4 frames
    env = FrameStack(env,4)

    # Allow back tracking that helps agents are not discouraged too heavily
    # from exploring backwards if there is no way to advance
    # head-on in the level.
    env = AllowBacktracking(env)

    return env

def make_train_0():
    return make_env(0)

def make_train_1():
    return make_env(1)

def make_train_2():
    return make_env(2)

def make_train_3():
    return make_env(3)

def make_train_4():
    return make_env(4)

def make_train_5():
    return make_env(5)

def make_train_6():
    return make_env(6)

def make_train_7():
    return make_env(7)

def make_train_8():
    return make_env(8)

def make_train_9():
    return make_env(9)

def make_train_10():
    return make_env(10)

def make_train_11():
    return make_env(11)

def make_train_12():
    return make_env(12)

def make_test_level_Green():
    return make_test()


def make_test():
    """
    Create an environment with some standard wrappers.
    """

    # Make the environment
    env = make_retro(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act2', record="./records")

    # Build the actions array
    env = ActionsDiscretizer(env)

    # Scale the rewards
    env = RewardScaler(env)

    # PreprocessFrame
    env = PreprocessFrame(env)

    # Stack 4 frames
    env = FrameStack(env, 4)

    # Allow back tracking that helps agents are not discouraged too heavily
    # from exploring backwards if there is no way to advance
    # head-on in the level.
    env = AllowBacktracking(env)

    return env









