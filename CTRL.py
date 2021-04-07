

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
from utils import *

def test():
    env = gym_super_mario_bros.make("SuperMarioBros-{}-{}-v0".format(1, 1))
    # if output_path:
    #     monitor = Monitor(256, 240, output_path)
    # else:
    #     monitor = None

    actions = COMPLEX_MOVEMENT
    env = JoypadSpace(env, actions)

    # env = CustomReward(env, monitor)
    #
    # env = CustomSkipFrame(env)
    tiles = SMB.get_tiles_num(env.unwrapped.ram)
    tiles = process_tiles(tiles)
    for i_episode in range(100):
        observation = env.reset()
        for t in range(1000000000):

            env.render()  # 更新动画
            if t < 20:
                action = int(1)
            else:
                action = int(4)
            # action = t
            observation, reward, done, info = env.step(action)  # 推进一步
            tiles = SMB.get_tiles_num(env.unwrapped.ram)
            tiles = process_tiles(tiles)
            if done:
                env.reset()
                continue


if __name__ == '__main__':
    test()