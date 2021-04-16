import timeit
from collections import deque

import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.distributions import Categorical

from src.env import create_train_env
from src.model_tiles import ActorCritic
from utils import *


def local_train(index, opt, global_model, optimizer, save=False):
    info = {"flag_get": False}
    torch.manual_seed(123 + index)
    die_id = 0
    if save:
        start_time = timeit.default_timer()

    writer = SummaryWriter(opt.log_path)

    env, num_states, num_actions = create_train_env(opt.world, opt.stage, opt.action_type)
    tiles = SMB.get_tiles_num(env.unwrapped.ram)
    tiles = process_tiles(tiles)

    local_model = ActorCritic(1, num_actions)

    if opt.use_gpu:
        local_model.cuda()

    local_model.train()
    env.reset()
    state = torch.from_numpy(tiles).unsqueeze(0).unsqueeze(0).float()

    if opt.use_gpu:
        state = state.cuda()

    done = True
    curr_step = 0
    curr_episode = 0

    while True:

        if save:
            print("Process {}. Episode {}".format(index, curr_episode))

        curr_episode += 1

        local_model.load_state_dict(global_model.state_dict())

        if done:
            h_0 = torch.zeros((1, 512), dtype=torch.float)
            c_0 = torch.zeros((1, 512), dtype=torch.float)
        else:
            h_0 = h_0.detach()
            c_0 = c_0.detach()
        if opt.use_gpu:
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()

        log_policies = []
        values = []
        rewards = []
        entropies = []

        for _ in range(opt.num_local_steps):

            curr_step += 1
            logits, value, h_0, c_0 = local_model(state, h_0, c_0)

            policy = F.softmax(logits, dim=1)
            log_policy = F.log_softmax(logits, dim=1)
            entropy = -(policy * log_policy).sum(1, keepdim=True)

            m = Categorical(policy)
            action = m.sample().item()

            if rules(tiles, action):
                done = True
                reward = -8
            else:
                state, reward, done, info = env.step(action)

                tiles = SMB.get_tiles_num(env.unwrapped.ram)
                tiles = process_tiles(tiles)
                state = torch.from_numpy(tiles).unsqueeze(0).unsqueeze(0).float()

                # env.render()

            if opt.use_gpu:
                state = state.cuda()
            if curr_step > opt.num_global_steps:
                done = True

            if done:
                die_id += 1
                print(die_id)
                curr_step = 0
                state = torch.from_numpy(tiles).unsqueeze(0).unsqueeze(0).float()
                env.reset()
                if opt.use_gpu:
                    state = state.cuda()

            values.append(value)
            log_policies.append(log_policy[0, action])
            rewards.append(reward)
            entropies.append(entropy)

            if done:
                break

        R = torch.zeros((1, 1), dtype=torch.float)

        if opt.use_gpu:
            R = R.cuda()
        if not done:
            _, R, _, _ = local_model(state, h_0, c_0)

        gae = torch.zeros((1, 1), dtype=torch.float)
        if opt.use_gpu:
            gae = gae.cuda()
        actor_loss = 0
        critic_loss = 0
        entropy_loss = 0
        next_value = R

        for value, log_policy, reward, entropy in list(zip(values, log_policies, rewards, entropies))[::-1]:
            gae = gae * opt.gamma * opt.tau
            gae = gae + reward + opt.gamma * next_value.detach() - value.detach()
            next_value = value
            actor_loss = actor_loss + log_policy * gae
            R = R * opt.gamma + reward
            critic_loss = critic_loss + (R - value) ** 2 / 2
            entropy_loss = entropy_loss + entropy

        total_loss = -actor_loss + critic_loss - opt.beta * entropy_loss

        writer.add_scalar("Train_{}/Loss".format(index), total_loss, curr_episode)
        optimizer.zero_grad()
        total_loss.backward()

        for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
            if global_param.grad is not None:
                break
            global_param._grad = local_param.grad

        optimizer.step()

        if curr_episode == int(opt.num_global_steps / opt.num_local_steps):
            print("Training process {} terminated".format(index))
            if save:
                end_time = timeit.default_timer()
                print('The code runs for %.2f s ' % (end_time - start_time))
            return

        if curr_episode % opt.save_interval == 0:
            # if info["flag_get"]:
            if local_test(opt.num_processes, opt, global_model, start_time, curr_episode):
                break


def local_test(index, opt, global_model, start_time, curr_episode):
    info = {"flag_get": False}
    torch.manual_seed(123 + index)
    env, num_states, num_actions = create_train_env(opt.world, opt.stage, opt.action_type)
    tiles = SMB.get_tiles_num(env.unwrapped.ram)
    tiles = process_tiles(tiles)
    local_model = ActorCritic(1, num_actions)

    local_model.eval()
    env.reset()
    state = torch.from_numpy(tiles).unsqueeze(0).unsqueeze(0).float()

    done = True
    curr_step = 0
    actions = deque(maxlen=opt.max_actions)
    while True and info["flag_get"] == False:
        curr_step += 1
        if done:
            local_model.load_state_dict(global_model.state_dict())
        with torch.no_grad():
            if done:
                h_0 = torch.zeros((1, 512), dtype=torch.float)
                c_0 = torch.zeros((1, 512), dtype=torch.float)
            else:
                h_0 = h_0.detach()
                c_0 = c_0.detach()

        logits, value, h_0, c_0 = local_model(state, h_0, c_0)
        policy = F.softmax(logits, dim=1)
        action = torch.argmax(policy).item()
        state, reward, done, info = env.step(action)
        tiles = SMB.get_tiles_num(env.unwrapped.ram)
        tiles = process_tiles(tiles)

        env.render()
        actions.append(action)

        if curr_step > opt.num_global_steps or actions.count(actions[0]) == actions.maxlen:
            done = True

        if done:
            curr_step = 0
            actions.clear()
            state = env.reset()
        state = torch.from_numpy(tiles).unsqueeze(0).unsqueeze(0).float()

        if info["flag_get"]:
            print("完成")
            end_time = timeit.default_timer()
            config_state = {'net': global_model.state_dict(),
                            'curr_episode': curr_episode,
                            'time': end_time - start_time,
                            }

            torch.save(config_state,
                       "{}/a3c_super_mario_bros_{}_{}".format(opt.saved_path, opt.world, opt.stage))

            return True
        else:
            env.close()
            return False


def rules(tiles, action):
    COMPLEX_MOVEMENT = [
        ['NOOP'],
        ['right'],
        ['right', 'A'],
        ['right', 'B'],
        ['right', 'A', 'B'],
        ['A'],
        ['left'],
        ['left', 'A'],
        ['left', 'B'],
        ['left', 'A', 'B'],
        ['down'],
        ['up'],
    ]

    if (action == 2 or action == 4) and 170 in tiles and np.argwhere(tiles == 170)[0, 0] == 12 and tiles[
        13, np.argwhere(tiles == 170)[0, 1]] == 0:
        return True
    else:
        return False
