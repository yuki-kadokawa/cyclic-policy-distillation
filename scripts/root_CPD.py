import numpy as np

from SAC.GlobalSACAgent import GlobalSACAgent
from SAC.LocalSACAgent import LocalSACAgent
from Environment.EnvironmentFactory import EnvironmentFactory
from UserDefinedSettings import UserDefinedSettings


class CyclicPolicyDistillation(object):
    def __init__(self):
        self.check_test = False
        self.check_global = False
        self.make_global_last_flag = True
        self.onPolicy_distillation = True

    def run(self):
        LEARNING_METHOD = 'CPD'
        userDefinedSettings = UserDefinedSettings(LEARNING_METHOD)
        environmentFactory = EnvironmentFactory(userDefinedSettings)
        self.userDefinedSettings = userDefinedSettings
        env = environmentFactory.generate()

        local_field_list = self.make_localField_domainRange(env, userDefinedSettings)

        global_field = GlobalSACAgent(env, userDefinedSettings, userDefinedSettings.DOMAIN_NUM)

        for rollout_times in range(99999):
            rollout_num = 15
            if rollout_times * rollout_num * 2 >= userDefinedSettings.rollout_cycle_num:
                break

            if self.check_global and (rollout_times * 2) % userDefinedSettings.check_global_interbal == 0 and rollout_times != 0:
                global_field.learn_and_evaluate_by_distillation(local_field_list, onPolicy=self.onPolicy_distillation)
            for learning_target_domain_num in range(len(local_field_list)):  # [0,1,2,...]
                print('DOMAIN: ', learning_target_domain_num)
                for _ in range(rollout_num):
                    self.set_learning_domain_value(local_field_list, learning_target_domain_num)
                    self.update(local_field_list, learning_target_domain_num, direction='forward')
                local_field_list[learning_target_domain_num].save_model()
                self.initialize_field(learning_target_domain_num, local_field_list, direction='forward')

            if self.check_global and (rollout_times * 2 + 1) % userDefinedSettings.check_global_interbal == 0:
                global_field.learn_and_evaluate_by_distillation(local_field_list, onPolicy=self.onPolicy_distillation)
            for learning_target_domain_num in range(len(local_field_list) - 1, -1, -1):  # [N,N-1,...]
                print('DOMAIN: ', learning_target_domain_num)
                for _ in range(rollout_num):
                    self.set_learning_domain_value(local_field_list, learning_target_domain_num)
                    self.update(local_field_list, learning_target_domain_num, direction='backward')
                local_field_list[learning_target_domain_num].save_model()
                self.initialize_field(learning_target_domain_num, local_field_list, direction='backward')
        if self.make_global_last_flag:
            global_field.learn_and_evaluate_by_distillation(local_field_list, onPolicy=self.onPolicy_distillation)
            global_field.save_model()

    def update(self, local_field_list, learning_target_domain_num, direction='forward'):
        if direction == 'forward':
            if learning_target_domain_num == 0:
                distillation_field = None
            else:
                distillation_field = local_field_list[learning_target_domain_num - 1]
        elif direction == 'backward':
            if learning_target_domain_num == len(local_field_list) - 1:
                distillation_field = None
            else:
                distillation_field = local_field_list[learning_target_domain_num + 1]

        local_field_list[learning_target_domain_num].rollout_and_update(
            distillation_field=distillation_field,
            distillation_field_learn_flag=False,
            sample_domain_num=learning_target_domain_num,
            distillation_field_get_sample_flag=False,
            learn_own_using_dist_flag=True,
            distillation_field_learn_in_RL_flag=False,
            distillation_from_teacher_sample_flag=False,
            check_distillation_policy=False,
            check_own_policy=self.check_test,
            distillation_update_num=1)

    def initialize_field(self, current_domain_num, local_field_list, direction):
        if direction == 'forward':
            if current_domain_num == len(local_field_list) - 1:
                return
            else:
                update_target_domain = current_domain_num + 1
        elif direction == 'backward':
            if current_domain_num == 0:
                return
            else:
                update_target_domain = current_domain_num - 1

        local_field_list[update_target_domain].critic.initialize_value_function_by_expert(local_field_list[current_domain_num].critic)

    def make_localField_domainRange(self, env, userDefinedSettings):
        local_field_list = []

        for current_domain_num in range(userDefinedSettings.DOMAIN_NUM):
            local_field = LocalSACAgent(env, userDefinedSettings)
            local_field.domain_num = current_domain_num
            local_field.set_summary_writer()
            local_field_list.append(local_field)

        return local_field_list

    def set_learning_domain_value(self, local_field_list, target_domain_num):
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
        local_field_list[target_domain_num].env.user_direct_set_domain_parameters(set_domains)


if __name__ == '__main__':
    cyclicPolicyDistillation = CyclicPolicyDistillation()
    cyclicPolicyDistillation.run()
