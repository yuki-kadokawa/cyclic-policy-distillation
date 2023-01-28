import os
import datetime
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--seed", help="seed", type=int, default=1)
parser.add_argument("--alpha", help="mixture rate type of policy distillation: -1->MI, 0-1->const.", type=float, default=-1.)
parser.add_argument("--gpu", help="gpu id", type=str, default='0')
args = parser.parse_args()


class UserDefinedSettings(object):

    def __init__(self, LEARNING_METHOD='_', BASE_RL_METHOD='SAC'):

        self.DEVICE = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
        self.ENVIRONMENT_NAME = 'Pendulum'
        current_time = datetime.datetime.now()
        file_name = 'M{:0=2}D{:0=2}H{:0=2}M{:0=2}S{:0=2}'.format(current_time.month, current_time.day, current_time.hour, current_time.minute, current_time.second)
        self.LOG_DIRECTORY = os.path.join('logs', self.ENVIRONMENT_NAME, LEARNING_METHOD + 'with' + BASE_RL_METHOD, file_name)
        self.LSTM_FLAG = True
        self.DOMAIN_RANDOMIZATION_FLAG = True
        self.BASE_RL_METHOD = BASE_RL_METHOD
        self.seed = args.seed

        self.num_steps = 1e6
        self.batch_size = 16
        self.policy_update_start_episode_num = 50
        self.learning_rate = self.lr = 1e-4
        self.HIDDEN_NUM = 64
        self.GLOBAL_DIST_ITERATION_NUM = 30
        self.DOMAIN_NUM = 4
        self.check_global_interbal = 1
        self.rollout_cycle_num = 240

        self.onPolicy_distillation = True

        self.memory_size = 1e6
        self.gamma = 0.99
        self.soft_update_rate = 0.005
        self.entropy_tuning = True
        self.entropy_tuning_scale = 1
        self.entropy_coefficient = 0.2
        self.multi_step_reward_num = 1
        self.updates_per_step = 1
        self.target_update_interval = 1
        self.evaluate_interval = 10
        self.initializer = 'xavier'
        self.run_num_per_evaluate = 3
        self.average_num_for_model_save = self.run_num_per_evaluate
        self.LEARNING_REWARD_SCALE = 1.
        self.MODEL_SAVE_INDEX = 'test'

        self.set_policy_mixture_rate = args.alpha
        self.value_init_flag = True
        self.policy_init_flag = False
