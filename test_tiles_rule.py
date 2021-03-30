"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""

import os

os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import torch
from src.env import create_train_env
from src.model_tiles import ActorCritic
import torch.nn.functional as F
from utils import *


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of model described in the paper: Asynchronous Methods for Deep Reinforcement Learning for Super Mario Bros""")
    parser.add_argument("--world", type=int, default=2)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--action_type", type=str, default="complex")
    parser.add_argument("--saved_path", type=str, default="trained_models_tiles_rule")
    parser.add_argument("--output_path", type=str, default="output")
    args = parser.parse_args()
    return args


def test(opt):
    torch.manual_seed(123)
    env, num_states, num_actions = create_train_env(opt.world, opt.stage, opt.action_type,
                                                    "{}/video_{}_{}.mp4".format(opt.output_path, opt.world, opt.stage))

    # env, num_states, num_actions = create_train_env(2, opt.stage, opt.action_type,
    #                                                 "{}/video_{}_{}.mp4".format(opt.output_path, 2, opt.stage))
    model = ActorCritic(1, num_actions)
    if torch.cuda.is_available():
        model_dict = torch.load("{}/a3c_super_mario_bros_{}_{}".format(opt.saved_path, opt.world, opt.stage))
        model.load_state_dict(model_dict['net'])
        model.cuda()
        print("episode", model_dict['curr_episode'])
        print("time", model_dict['time'])
    else:
        model_dict = torch.load("{}/a3c_super_mario_bros_{}_{}".format(opt.saved_path, opt.world, opt.stage),
                                map_location=lambda storage, loc: storage)
        model.load_state_dict(model_dict['net'])
        print("episode", model_dict['curr_episode'])
        print("time", model_dict['time'])

    model.eval()
    env.reset()
    tiles = SMB.get_tiles_num(env.unwrapped.ram)
    tiles = process_tiles(tiles)
    state = torch.from_numpy(tiles).unsqueeze(0).unsqueeze(0).float()
    done = True
    while True:
        if done:
            h_0 = torch.zeros((1, 512), dtype=torch.float)
            c_0 = torch.zeros((1, 512), dtype=torch.float)
            env.reset()
        else:
            h_0 = h_0.detach()
            c_0 = c_0.detach()
        if torch.cuda.is_available():
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()
            state = state.cuda()

        logits, value, h_0, c_0 = model(state, h_0, c_0)
        policy = F.softmax(logits, dim=1)
        action = torch.argmax(policy).item()
        action = int(action)
        state, reward, done, info = env.step(action)
        print(reward)
        # print(reward, done, action)
        tiles = SMB.get_tiles_num(env.unwrapped.ram)
        tiles = process_tiles(tiles)
        state = torch.from_numpy(tiles).unsqueeze(0).unsqueeze(0).float()
        # print(done,info["flag_get"])
        print(reward)
        env.render()
        if info["flag_get"]:
            print("World {} stage {} completed".format(opt.world, opt.stage))
            break


if __name__ == "__main__":
    opt = get_args()
    test(opt)
