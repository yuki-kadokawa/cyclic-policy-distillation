import numpy as np

from SAC.LocalSACAgent import LocalSACAgent


class GlobalSACAgent(object):
    def __init__(self, env, userDefinedSettings, domain_num=None):
        self.global_field = LocalSACAgent(env, userDefinedSettings)
        self.userDefinedSettings = userDefinedSettings

        self.global_field.domain_num = 999
        self.global_field.set_summary_writer()
        self.summary_counter = 0
        self.global_distillation_counter = 0

        self.episode_length = self.global_field.env.MAX_EPISODE_LENGTH
        self.domain_num = domain_num

    def save_model(self):
        self.global_field.save_model()

    def learn_and_evaluate_by_distillation(self, local_field_list, onPolicy=True):
        if onPolicy:
            self.distillation_global(local_field_list, onPolicy=onPolicy)
        else:
            self.distillation_global_offPolicy(local_field_list)

    def distillation_global(self, local_field_list, onPolicy=True):
        sample_episode_num_per_cycle = 5
        update_iteration_num = self.userDefinedSettings.GLOBAL_DIST_ITERATION_NUM
        update_num = 1000
        batch_size = 16
        test_num_per_each_domain = 5

        self.global_field.replay_buffer.clear()
        self.global_field.actor.initialize_policy()

        for iteration_count in range(int(update_iteration_num / sample_episode_num_per_cycle)):
            print('iteration_count: {} / {}'.format(iteration_count * sample_episode_num_per_cycle, update_iteration_num))
            self.make_distillation_dataset(local_field_list, sampling_policy='global', sample_episode_num=sample_episode_num_per_cycle, onPolicy=onPolicy)
            self.update_global_policy(update_num=update_num, batch_size=batch_size)
        if True:
            global_domain_reward_list, _ = self.evaluate_global_policy_local_domain(test_num_per_each_domain=test_num_per_each_domain, save_sammary_flag=False)
            self.global_field.summaryWriter.add_scalars('status/reward all', {'global': np.average(global_domain_reward_list)}, self.global_distillation_counter)
            local_domain_reward_list, _ = self.evaluate_local_policies(local_field_list, local_test_num_per_each_domain=test_num_per_each_domain, save_sammary_flag=False)
            self.global_field.summaryWriter.add_scalars('status/reward all', {'local': np.average(local_domain_reward_list)}, self.global_distillation_counter)
            print(np.average(global_domain_reward_list), np.average(local_domain_reward_list))
        self.global_distillation_counter += 1

    def distillation_global_offPolicy(self, local_field_list):
        sample_episode_num_per_cycle = 5
        update_iteration_num = self.userDefinedSettings.GLOBAL_DIST_ITERATION_NUM
        update_num = 10 * (int(update_iteration_num / sample_episode_num_per_cycle))
        batch_size = 16
        test_num_per_each_domain = 5

        self.global_field.replay_buffer.clear()
        self.global_field.actor.initialize_policy()

        if len(local_field_list[0].replay_buffer) > batch_size:
            self.global_field.replay_buffer.get_marge(local_field_list)
            self.update_global_policy(update_num=update_num, batch_size=batch_size)

        if True:
            global_domain_reward_list, _ = self.evaluate_global_policy_local_domain(test_num_per_each_domain=test_num_per_each_domain, save_sammary_flag=False)
            self.global_field.summaryWriter.add_scalars('status/reward all', {'global': np.average(global_domain_reward_list)}, self.global_distillation_counter)
        self.global_distillation_counter += 1

    def make_distillation_dataset(self, local_field_list, sampling_policy='local', sample_episode_num=None, onPolicy=True):
        for domain_num, local_field in enumerate(local_field_list):
            for _ in range(sample_episode_num):
                self.set_learning_domain_value(local_field, domain_num)
                local_field.sample_dataset(global_field=self.global_field, sample_episode_num=1, sampling_policy=sampling_policy, onPolicy=onPolicy)

    def update_global_policy(self, update_num=None, batch_size=None):
        self.global_field.update_supervised(update_num=update_num, batch_size=batch_size)

    def evaluate_global_policy_random_domain(self):
        self.set_learning_domain_range(field=self.global_field)
        test_avarage_reward, task_achievement_rate = self.global_field.test(test_num=self.global_test_num_per_each_domain, render_flag=False, reward_show_flag=False)
        self.global_field.summaryWriter.add_scalar('status/murged policy reward', test_avarage_reward, self.summary_counter)
        self.global_field.summaryWriter.add_scalar('status/murged policy task achievement', task_achievement_rate, self.summary_counter)
        print('R:{}, A:{}'.format(test_avarage_reward, task_achievement_rate))
        self.summary_counter += 1

    def evaluate_global_policy_local_domain(self, iteration_count=None, test_num_per_each_domain=None, save_sammary_flag=False):
        domain_reward_list = []
        domain_achieve_list = []
        for domain_num in range(self.domain_num):
            reward_list = []
            achieve_list = []
            for test_num in range(test_num_per_each_domain):
                self.set_learning_domain_value(self.global_field, domain_num)
                print('test')
                test_avarage_reward, task_achievement_rate = self.global_field.test(test_num=1, render_flag=False, reward_show_flag=False)
                if save_sammary_flag is True:
                    counter = self.global_distillation_counter * self.total_rollout_episode_num + iteration_count * test_num_per_each_domain + test_num
                    self.global_field.summaryWriter.add_scalars('status/reward' + str(domain_num), {'global': test_avarage_reward}, counter)
                    self.global_field.summaryWriter.add_scalars('status/achieve' + str(domain_num), {'global': task_achievement_rate}, counter)
                reward_list.append(test_avarage_reward)
                achieve_list.append(task_achievement_rate)
            domain_reward_list.append(np.average(reward_list))
            domain_achieve_list.append(np.average(achieve_list))
        return domain_reward_list, domain_achieve_list

    def evaluate_local_policies(self, local_field_list, local_test_num_per_each_domain, save_sammary_flag=False):
        domain_reward_list = []
        domain_achieve_list = []
        for domain_num, local_field in zip(range(len(local_field_list)), local_field_list):
            reward_list = []
            achieve_list = []
            for test_num in range(local_test_num_per_each_domain):
                print(test_num)
                self.set_learning_domain_value(local_field, domain_num)
                test_avarage_reward, task_achievement_rate = local_field.test(test_num=1, render_flag=False, reward_show_flag=False)
                if save_sammary_flag is True:
                    self.global_field.summaryWriter.add_scalars('status/reward' + str(domain_num), {'local': test_avarage_reward}, test_num)
                    self.global_field.summaryWriter.add_scalars('status/achieve' + str(domain_num), {'local': task_achievement_rate}, test_num)
                reward_list.append(test_avarage_reward)
                achieve_list.append(task_achievement_rate)
            domain_reward_list.append(np.average(reward_list))
            domain_achieve_list.append(np.average(achieve_list))
        return domain_reward_list, domain_achieve_list

    def set_learning_domain_value(self, local_field, target_domain_num):
        all_domain_num = float(self.userDefinedSettings.DOMAIN_NUM)
        if target_domain_num < all_domain_num / 2:
            first_domain_range_min = 0.
            first_domain_range_max = float((target_domain_num + 1.)) / (all_domain_num / 2.)
        else:
            first_domain_range_min = (target_domain_num - (all_domain_num / 2.)) / (all_domain_num / 2.)
            first_domain_range_max = 1.
        first_domain_value = (first_domain_range_max - first_domain_range_min) * np.random.rand() + first_domain_range_min

        min_bias = float((target_domain_num + 0.) / (all_domain_num / 2.))
        max_bias = float((target_domain_num + 1.) / (all_domain_num / 2.))
        second_domain_range_min = max(-1. * first_domain_value + min_bias, 0.)
        second_domain_range_max = min(-1. * first_domain_value + max_bias, 1)
        second_domain_value = (second_domain_range_max - second_domain_range_min) * np.random.rand() + second_domain_range_min
        set_domains = [first_domain_value, second_domain_value]
        local_field.env.user_direct_set_domain_parameters(set_domains)
