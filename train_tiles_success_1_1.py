"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import os  # NOQA: E402

os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # NOQA: E402

import argparse
import torch
from src.env import create_train_env
from src.model_tiles import ActorCritic
from src.optimizer import GlobalAdam
from src.process_tiles_success import local_train, local_test
import torch.multiprocessing as _mp
import shutil


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of model described in the paper: Asynchronous Methods for Deep Reinforcement Learning for 
        Super Mario Bros""")
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=2)
    parser.add_argument("--action_type", type=str, default="complex")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.9, help='discount factor for rewards')
    parser.add_argument('--tau', type=float, default=1.0, help='parameter for GAE')
    parser.add_argument('--beta', type=float, default=0.01, help='entropy coefficient')
    parser.add_argument("--num_local_steps", type=int, default=50)
    parser.add_argument("--num_global_steps", type=int, default=5e6)
    parser.add_argument("--num_processes", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=500, help="Number of steps between savings")
    parser.add_argument("--max_actions", type=int, default=200, help="Maximum repetition steps in test phase")
    parser.add_argument("--log_path", type=str, default="tensorboard/a3c_super_mario_bros_tiles_1")
    parser.add_argument("--saved_path", type=str, default="trained_models_tiles_1")
    parser.add_argument("--load_from_previous_stage", type=bool, default=True,
                        help="Load weight from previous trained stage")
    parser.add_argument("--use_gpu", type=bool, default=True)
    args = parser.parse_args()
    return args


def train(opt):
    torch.manual_seed(123)

    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)

    os.makedirs(opt.log_path)

    if not os.path.isdir(opt.saved_path):
        os.makedirs(opt.saved_path)

    mp = _mp.get_context("spawn")
    env, num_states, num_actions = create_train_env(opt.world, opt.stage, opt.action_type)
    global_model = ActorCritic(1, num_actions)

    if opt.use_gpu:
        global_model.cuda()

    global_model.share_memory()

    if opt.load_from_previous_stage:
        if opt.stage == 1:
            previous_world = opt.world - 1
            previous_stage = 4
        else:
            previous_world = opt.world
            previous_stage = opt.stage - 1
        file_ = "{}/a3c_super_mario_bros_{}_{}".format("trained_models_tiles", "1", "1")
        if os.path.isfile(file_):
            model_dict = torch.load(file_)
            global_model.load_state_dict(model_dict['net'])

    optimizer = GlobalAdam(global_model.parameters(), lr=opt.lr)
    local_train(0, opt, global_model, optimizer, True)


if __name__ == "__main__":
    opt = get_args()
    train(opt)
