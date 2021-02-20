import maml_rl.envs
import gym
import torch
import numpy as np
# from tqdm import trange
import yaml

from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.samplers import MultiTaskSampler
from maml_rl.utils.helpers import get_policy_for_env, get_input_size
from maml_rl.utils.reinforcement_learning import get_returns


def main(args):
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    if args.output_folder is not None:
        if not os.path.exists(args.output_folder):
            os.makedirs(args.output_folder, exist_ok=True)
        logs_filename = os.path.join(args.output_folder, 'logs_eval')

    env = gym.make(config['env-name'], **config['env-kwargs'])
    env.close()

    # Policy
    policy = get_policy_for_env(env,
                                hidden_sizes=config['hidden-sizes'],
                                nonlinearity=config['nonlinearity'])
    with open(args.policy, 'rb') as f:
        state_dict = torch.load(f, map_location=torch.device(args.device))
        policy.load_state_dict(state_dict)
    policy.share_memory()

    # Baseline
    baseline = LinearFeatureBaseline(get_input_size(env))

    # Sampler
    sampler = MultiTaskSampler(config['env-name'],
                               env_kwargs=config['env-kwargs'],
                               batch_size=args.fast_batch_size,
                               policy=policy,
                               baseline=baseline,
                               env=env,
                               seed=args.seed,
                               num_workers=args.num_workers)

    logs = {'tasks': []}
    train_returns, valid_returns = [], []
    for i in range(args.num_steps):
        exec("after_%s_gradient_step=[]" % i)

    for batch in range(args.num_batches):
        tasks = sampler.sample_tasks(num_tasks=args.meta_batch_size)
        '''
        train_episodes is a list whose length is the number of gradient steps, 
        and each element is also a list of length meta_batch_size containing the different episodes. 
        For example, train_episodes[0] contains the episodes before any gradient update, 
        train_episodes[1] the episodes after 1 gradient update (if the number of steps of adaptation is > 1), and so on.
        
        valid_episodes is a list containing the episodes after all the steps of adaptation.
        
        
        
        MultiTaskSampler, which is responsible for sampling the trajectories, is doing adaptation locally in each worker.
        
        from line 270 to line 275 in multi_task_sampler.py:
        
        with self.policy_lock: 
        loss = reinforce_loss(self.policy, train_episodes, params=params) 
        params = self.policy.update_params(loss, 
                                           params=params, 
                                           step_size=fast_lr, 
                                           first_order=True) 
        
        So in test.py, you do get both trajectories before and after adaptation with the simple call to MultiTaskSampler. 
        And with a few changes to test.py you can even use different number of gradient steps for adaptation by changing 
        num_steps in your call to sampler.sample().
        '''
        train_episodes, valid_episodes = sampler.sample(tasks,
                                                        num_steps=args.num_steps,
                                                        fast_lr=config['fast-lr'],
                                                        gamma=config['gamma'],
                                                        gae_lambda=config['gae-lambda'],
                                                        device=args.device)

        logs['tasks'].extend(tasks)
        train_returns.append(get_returns(train_episodes[0]))
        valid_returns.append(get_returns(valid_episodes))

        for i in range(args.num_steps):
            exec("after_%s_gradient_step.append(get_returns(train_episodes[%i]))" % (i,i))


        logs['train_returns'] = np.concatenate(train_returns, axis=0)
        logs['valid_returns'] = np.concatenate(valid_returns, axis=0)

        for i in range(args.num_steps):
            exec("logs['after_%s_gradient_step'] = np.concatenate(after_%s_gradient_step, axis=0)" % (i,i))

        with open(logs_filename, 'wb') as f:
            np.savez(f, **logs)

        print('batch: ', batch)


if __name__ == '__main__':
    import argparse
    import os
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='Reinforcement learning with '
        'Model-Agnostic Meta-Learning (MAML) - Test')

    parser.add_argument('--config', type=str, required=True,
        help='path to the configuration file')
    parser.add_argument('--policy', type=str, required=True,
        help='path to the policy checkpoint')

    # Evaluation
    evaluation = parser.add_argument_group('Evaluation')
    evaluation.add_argument('--num-batches', type=int, default=1,
        help='number of batches (default: 1)')
    evaluation.add_argument('--meta-batch-size', type=int, default=100,
        help='number of tasks per batch (default: 100)')
    evaluation.add_argument('--fast-batch-size', type=int, default=1,
                            help='Number of trajectories to sample for each task (default: 1)')
    evaluation.add_argument('--num-steps', type=int, default=10,
                            help='Number of gradient steps in the inner loop / fast adaptation (default: 10)')

    '''
    --num-batches: total evaluation tasks batches
    --meta-batch-size: number of tasks per batch
    --fast-batch-size: Number of trajectories to sample for each task
    --num-steps: Number of gradient steps in the inner loop / fast adaptation
    '''

    # Miscellaneous
    misc = parser.add_argument_group('Miscellaneous')
    misc.add_argument('--output-folder', type=str, required=True,
        help='name of the output folder (default: maml)')
    misc.add_argument('--seed', type=int, default=1,
        help='random seed (default: 1)')
    misc.add_argument('--num-workers', type=int, default=mp.cpu_count() - 1,
        help='number of workers for trajectories sampling (default: '
             '{0})'.format(mp.cpu_count() - 1))
    misc.add_argument('--use-cuda', action='store_true',
        help='use cuda (default: false, use cpu). WARNING: Full upport for cuda '
        'is not guaranteed. Using CPU is encouraged.')

    args = parser.parse_args()
    args.device = ('cuda' if (torch.cuda.is_available()
                   and args.use_cuda) else 'cpu')

    main(args)
