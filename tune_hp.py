import gym
import torch
import json
import numpy as np
import time
import os
import yaml
from tqdm import trange

import maml_rl.envs
from maml_rl.metalearners import MAMLTRPO
from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.samplers import MultiTaskSampler
from maml_rl.utils.helpers import get_policy_for_env, get_input_size
from maml_rl.utils.reinforcement_learning import get_returns
import optuna
import warnings

warnings.filterwarnings("ignore")

def objective(tri):
    par = {
        'nonlinearity': tri.suggest_categorical('nonlinearity', ['tanh', 'leaky_relu']),

        'first-order': tri.suggest_categorical('first-order', ['true', 'false']),
        'fast-lr': tri.suggest_loguniform('fast-lr', 1e-3, 1),
        'fast-batch-size': tri.suggest_int('fast-batch-size', 10, 100, step=10),

        'meta-batch-size': tri.suggest_int('meta-batch-size', 10, 100, step=10),
        'max-kl': tri.suggest_loguniform('max-kl', 1e-3, 1)

    }
    return train(par)

def train(par):
    import argparse
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='Reinforcement learning with '
                                                 'Model-Agnostic Meta-Learning (MAML) - Train')

    parser.add_argument('--config', type=str, required=True,
                        help='path to the configuration file.')

    # Miscellaneous
    misc = parser.add_argument_group('Miscellaneous')

    misc.add_argument('--seed', type=int, default=None,
                      help='random seed')
    misc.add_argument('--num-workers', type=int, default=mp.cpu_count() - 1,
                      help='number of workers for trajectories sampling (default: '
                           '{0})'.format(mp.cpu_count() - 1))
    misc.add_argument('--use-cuda', action='store_true',
                      help='use cuda (default: false, use cpu). WARNING: Full upport for cuda '
                           'is not guaranteed. Using CPU is encouraged.')

    args = parser.parse_args()
    args.device = ('cuda' if (torch.cuda.is_available()
                              and args.use_cuda) else 'cpu')

    return main(args, par)

def main(args, par):

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config.update(par)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    env = gym.make(config['env-name'], **config['env-kwargs'])
    env.close()

    # Policy
    policy = get_policy_for_env(env,
                                hidden_sizes=config['hidden-sizes'],
                                nonlinearity=config['nonlinearity'])
    policy.share_memory()

    # Baseline
    baseline = LinearFeatureBaseline(get_input_size(env))

    # Sampler
    sampler = MultiTaskSampler(config['env-name'],
                               env_kwargs=config['env-kwargs'],
                               batch_size=config['fast-batch-size'],
                               policy=policy,
                               baseline=baseline,
                               env=env,
                               seed=args.seed,
                               num_workers=args.num_workers)

    metalearner = MAMLTRPO(policy,
                           fast_lr=config['fast-lr'],
                           first_order=config['first-order'],
                           device=args.device)

    num_iterations = 0
    logs = {}

    for batch in range(config['num-batches']):
        tasks = sampler.sample_tasks(num_tasks=config['meta-batch-size'])
        futures = sampler.sample_async(tasks,
                                       num_steps=config['num-steps'],
                                       fast_lr=config['fast-lr'],
                                       gamma=config['gamma'],
                                       gae_lambda=config['gae-lambda'],
                                       device=args.device)
        # logs: diction
        logs = metalearner.step(*futures,
                                max_kl=config['max-kl'],
                                cg_iters=config['cg-iters'],
                                cg_damping=config['cg-damping'],
                                ls_max_steps=config['ls-max-steps'],
                                ls_backtrack_ratio=config['ls-backtrack-ratio'])

        '''
        train_episodes is a list whose length is the number of gradient steps, 
        and each element is also a list of length meta_batch_size containing the different episodes. 
        For example, train_episodes[0] contains the episodes before any gradient update, 
        train_episodes[1] the episodes after 1 gradient update (if the number of steps of adaptation is > 1), and so on.

        valid_episodes is a list containing the episodes after all the steps of adaptation.
        '''
        train_episodes, valid_episodes = sampler.sample_wait(futures)
        num_iterations += sum(sum(episode.lengths) for episode in train_episodes[0])
        num_iterations += sum(sum(episode.lengths) for episode in valid_episodes)
        # batch is training process
        logs.update(batch=batch,
                    tasks=tasks,
                    num_iterations=num_iterations,
                    train_returns=get_returns(train_episodes[0]),
                    valid_returns=get_returns(valid_episodes))

    train_returns = np.average(logs['train_returns'], axis=1)
    train_returns = np.average(train_returns)
    valid_returns = np.average(logs['valid_returns'], axis=1)
    valid_returns = np.average(valid_returns)

    return valid_returns + (valid_returns - train_returns)

if __name__=='__main__':
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
