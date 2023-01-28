import torch
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from DatasetHandler.ReplayMemory import ReplayMemory
from .Actor.ActorBasic import ActorBasic
from .Critic.CriticBasic import CriticBasic
from .Critic.CriticLSTM import CriticLSTM
from .Actor.ActorLSTM import ActorLSTM
from .EntropyTerm.EntropyTerm import EntropyTerm
from LearningCommonParts.TotalRewardService import TotalRewardService, ProgressChecker


class LocalSACAgent(object):
    def __init__(self, env, userDefinedSettings):
        self.env = env
        self.userDefinedSettings = userDefinedSettings
        self.replay_buffer = ReplayMemory(env.STATE_DIM, env.ACTION_DIM, env.MAX_EPISODE_LENGTH, env.DOMAIN_PARAMETER_DIM, userDefinedSettings)
        if userDefinedSettings.LSTM_FLAG:
            self.critic = CriticLSTM(env.STATE_DIM, env.ACTION_DIM, env.DOMAIN_PARAMETER_DIM, userDefinedSettings)
            self.actor = ActorLSTM(env.STATE_DIM, env.ACTION_DIM, userDefinedSettings)
        else:
            self.critic = CriticBasic(env.STATE_DIM, env.ACTION_DIM, env.DOMAIN_PARAMETER_DIM, userDefinedSettings)
            self.actor = ActorBasic(env.STATE_DIM, env.ACTION_DIM, userDefinedSettings)
        self.entropyTerm = EntropyTerm(env.ACTION_DIM, userDefinedSettings)
        self.totalRewardService = TotalRewardService(self.userDefinedSettings)
        self.taskAchievementService = ProgressChecker(self.userDefinedSettings)
        self.taskAchievementService_distillation = ProgressChecker(self.userDefinedSettings)
        self.summary_writer_count = 0

        self.rollout_num = 0
        self.domain_num = None

    def train(self, domain_num=None, expert_value_function=None, expert_policy=None):
        self.model_dir = os.path.join(self.userDefinedSettings.LOG_DIRECTORY, 'model', str(domain_num))
        self.summary_dir = os.path.join(self.userDefinedSettings.LOG_DIRECTORY, 'summary', str(domain_num))

        self.summaryWriter = SummaryWriter(log_dir=self.summary_dir)
        self.summary_writer_count = 0

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)

        if domain_num is None:
            total_episode_num = self.userDefinedSettings.learning_episode_num_all_domain
        else:
            total_episode_num = self.userDefinedSettings.learning_episode_num

        total_step_num = 0
        for episode_num in range(total_episode_num):

            if episode_num < self.userDefinedSettings.expert_policy_explor_episode_num and expert_policy is not None:
                get_action_fn = expert_policy.get_action
            else:
                get_action_fn = self.actor.get_action

            state = self.env.reset()

            total_reward = 0.
            for step_num in range(self.env.MAX_EPISODE_LENGTH):
                policy_update_flag = self.is_update(episode_num)
                action, method_depend_info = get_action_fn(state, step=step_num, deterministic=False, random_action_flag=not policy_update_flag)
                next_state, reward, done, domain_parameter, task_achievement = self.env.step(action, get_task_achievement=True)

                self.replay_buffer.push(state, action, reward, next_state, done, method_depend_info, domain_parameter=domain_parameter, step=step_num)

                state = next_state
                total_reward += reward

                if policy_update_flag:
                    for _ in range(self.userDefinedSettings.updates_per_step):
                        self.update(self.userDefinedSettings.batch_size, expert_value_function=expert_value_function, episode_num=episode_num)

                total_step_num += 1
                if done:
                    break

            test_avarage_reward = self.test(test_num=self.userDefinedSettings.run_num_per_evaluate, render_flag=False, reward_show_flag=False)
            if self.userDefinedSettings.MODEL_SAVE_INDEX == 'train':
                is_achieve_peak = self.totalRewardService.trainPeakChecker.append_and_check(total_reward)
            else:
                is_achieve_peak = self.totalRewardService.testPeakChecker.append_and_check(test_avarage_reward)
            if is_achieve_peak:
                print('Episode: {:>5} | Episode Reward: {:>8.2f} | Test Reward: {:>8.2f} | model updated!!'.format(episode_num, total_reward, test_avarage_reward))
                self.save_model()
            else:
                print('Episode: {:>5} | Episode Reward: {:>8.2f} | Test Reward: {:>8.2f} | model keep'.format(episode_num, total_reward, test_avarage_reward))
            self.summaryWriter.add_scalar('status/train reward', total_reward, episode_num)
            self.summaryWriter.add_scalar('status/test reward', test_avarage_reward, episode_num)

    def DistRL_train(self, domain_num=None, teacher_list=None, expert=None, expert_learn_flag=False, local_learn_flag=True, initial_action_random_flag=True, use_other_domain_flag=False):
        self.model_dir = os.path.join(self.userDefinedSettings.LOG_DIRECTORY, 'model', str(domain_num))
        self.summary_dir = os.path.join(self.userDefinedSettings.LOG_DIRECTORY, 'summary', str(domain_num))

        self.summaryWriter = SummaryWriter(log_dir=self.summary_dir)
        self.summary_writer_count = 0

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)

        for episode_num in range(self.userDefinedSettings.learning_episode_num):

            state = self.env.reset()

            total_reward = 0.
            for step_num in range(self.env.MAX_EPISODE_LENGTH):
                policy_update_flag = self.is_update(episode_num)
                random_action_flag = initial_action_random_flag and (not policy_update_flag)
                action, method_depend_info = self.actor.get_action(state, step=step_num, deterministic=False, random_action_flag=random_action_flag)
                next_state, reward, done, domain_parameter, task_achievement = self.env.step(action, get_task_achievement=True)

                self.replay_buffer.push(state, action, reward, next_state, done, method_depend_info, domain_parameter=domain_parameter, step=step_num)
                if use_other_domain_flag is True:
                    expert.replay_buffer.push(state, action, reward, next_state, done, method_depend_info, domain_parameter=domain_parameter, step=step_num)

                state = next_state
                total_reward += reward

                if policy_update_flag:
                    for _ in range(self.userDefinedSettings.updates_per_step):
                        if local_learn_flag is True:
                            self.DistRL_update(self.userDefinedSettings.batch_size, expert=expert)
                        if expert_learn_flag is True:
                            self.expert_DistRL_update(self.userDefinedSettings.batch_size, summaryWriter=self.summaryWriter, summary_writer_count=self.summary_writer_count, expert=expert)
                            if use_other_domain_flag is True:
                                expert.expert_DistRL_update(self.userDefinedSettings.batch_size, summaryWriter=self.summaryWriter, summary_writer_count=self.summary_writer_count, expert=expert)
                if done:
                    break

            if expert_learn_flag is False:
                test_avarage_reward, task_achievement_rate = self.test(test_num=self.userDefinedSettings.run_num_per_evaluate, render_flag=False, reward_show_flag=False)
            else:
                test_avarage_reward, task_achievement_rate = self.test(test_num=self.userDefinedSettings.run_num_per_evaluate, render_flag=False, reward_show_flag=False, policy=expert.actor)

            print('Episode: {:>5} | Episode Reward: {:>8.2f} | Test Reward: {:>8.2f}'.format(episode_num, total_reward, test_avarage_reward))
            self.summaryWriter.add_scalar('status/train reward', total_reward, episode_num)
            self.summaryWriter.add_scalar('status/test reward', test_avarage_reward, episode_num)

            average_task_achievement_rate = self.taskAchievementService.append_and_value(task_achievement_rate)
            self.summaryWriter.add_scalar('status/task achievement', average_task_achievement_rate, episode_num)

            if (average_task_achievement_rate == 1.0) and (episode_num > 10):
                self.save_model()
                break

        return episode_num

    def set_summary_writer(self):
        self.model_dir = os.path.join(self.userDefinedSettings.LOG_DIRECTORY, 'model', str(self.domain_num))
        self.summary_dir = os.path.join(self.userDefinedSettings.LOG_DIRECTORY, 'summary', str(self.domain_num))
        self.summaryWriter = SummaryWriter(log_dir=self.summary_dir)
        self.summary_writer_count = 0
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)

    def rollout_and_update(self,
                           distillation_field=None,
                           distillation_field_learn_flag=True,
                           distillation_field_learn_in_RL_flag=True,
                           initial_action_random_flag=True,
                           distillation_field_get_sample_flag=True,
                           learn_own_field_flag=True,
                           learn_own_using_dist_flag=True,
                           sample_domain_num=999,
                           multi_expert_flag=False,
                           distillation_from_teacher_sample_flag=False,
                           check_distillation_policy=False,
                           check_own_policy=True,
                           distillation_update_num=1
                           ):

        total_reward = 0.
        state = self.env.reset()
        for step_num in range(self.env.MAX_EPISODE_LENGTH):
            policy_update_flag = self.is_update(self.rollout_num)
            random_action_flag = initial_action_random_flag and (not policy_update_flag)
            action, method_depend_info = self.actor.get_action(state, step=step_num, deterministic=False, random_action_flag=random_action_flag)
            next_state, reward, done, domain_parameter, task_achievement = self.env.step(action, get_task_achievement=True)
            self.replay_buffer.push(state, action, reward, next_state, done, method_depend_info, domain_parameter=domain_parameter, step=step_num, debug_term=sample_domain_num)
            if (distillation_field is not None) and (distillation_field_get_sample_flag is True):
                if multi_expert_flag is True:
                    distillation_field[sample_domain_num].replay_buffer.push(state, action, reward, next_state, done, method_depend_info, domain_parameter=domain_parameter, step=step_num, debug_term=sample_domain_num)
                else:
                    distillation_field.replay_buffer.push(state, action, reward, next_state, done, method_depend_info, domain_parameter=domain_parameter, step=step_num, debug_term=sample_domain_num)

            state = next_state
            total_reward += reward

            if policy_update_flag:
                if learn_own_field_flag is True:
                    if learn_own_using_dist_flag is True:
                        self.DistRL_update(self.userDefinedSettings.batch_size, expert=distillation_field, multi_expert_flag=multi_expert_flag, distillation_update_num=distillation_update_num)
                    else:
                        self.DistRL_update(self.userDefinedSettings.batch_size)
                if (distillation_field is not None) and (distillation_field_learn_flag is True):
                    distillation_field.DistRL_update(self.userDefinedSettings.batch_size, expert=self,
                                                     learn_in_RL_flag=distillation_field_learn_in_RL_flag,
                                                     distillation_from_teacher_sample_flag=distillation_from_teacher_sample_flag)
            if done:
                break

        self.summaryWriter.add_scalar('status/train reward', total_reward, self.rollout_num)
        if check_own_policy is True:
            test_avarage_reward, _ = self.test(test_num=self.userDefinedSettings.run_num_per_evaluate, render_flag=False, reward_show_flag=False)
            print('Episode: {:>5} | Episode Reward: {:>8.2f} | Test Reward: {:>8.2f}'.format(self.rollout_num, total_reward, test_avarage_reward))
            self.summaryWriter.add_scalar('status/test reward current policy', test_avarage_reward, self.rollout_num)

        self.rollout_num += 1

        print('Episode: {:>5} | Episode Reward: {:>8.2f}'.format(self.rollout_num, total_reward))

    def sample_dataset(self, global_field, sample_episode_num=1, sampling_policy='local', onPolicy=True):
        if onPolicy:
            for _ in range(sample_episode_num):
                total_reward = 0.
                state = self.env.reset()
                for step_num in range(self.env.MAX_EPISODE_LENGTH):
                    global_action, global_method_depend_info = global_field.actor.get_action(state, step=step_num, deterministic=True, random_action_flag=False)
                    local_action, local_method_depend_info = self.actor.get_action(state, step=step_num, deterministic=True, random_action_flag=False)
                    if sampling_policy == 'global':
                        sampling_action = global_action
                        learing_target_action = local_action
                        method_depend_info = global_method_depend_info
                    elif sampling_policy == 'local':
                        sampling_action = local_action
                        learing_target_action = local_action
                        method_depend_info = local_method_depend_info

                    next_state, reward, done, domain_parameter = self.env.step(sampling_action, get_task_achievement=False)
                    global_field.replay_buffer.push(state, learing_target_action, reward, next_state, done, method_depend_info, domain_parameter=domain_parameter, step=step_num)
                    total_reward += reward
                    if done:
                        break
                    state = next_state
        else:
            for _ in range(sample_episode_num):
                batch = self.replay_buffer.sample(batch_size=1)
                for step_num in range(self.env.MAX_EPISODE_LENGTH):
                    global_field.replay_buffer.push(state, learing_target_action, reward, next_state, done, method_depend_info, domain_parameter=domain_parameter, step=step_num)

    def is_update(self, episode_num=99999):
        return len(self.replay_buffer) > self.userDefinedSettings.batch_size and episode_num > self.userDefinedSettings.policy_update_start_episode_num

    def update(self, batch_size, expert_value_function=None, episode_num=None):
        if self.userDefinedSettings.LSTM_FLAG:
            self.update_lstm(batch_size, expert_value_function=expert_value_function, episode_num=episode_num)
        else:
            self.update_basic(batch_size, expert_value_function=expert_value_function)

    def DistRL_update(self, batch_size, expert=None, multi_expert_flag=False, learn_in_RL_flag=True, distillation_from_teacher_sample_flag=False, distillation_update_num=1):
        self.DistRL_update_lstm(batch_size, expert=expert, multi_expert_flag=multi_expert_flag, learn_in_RL_flag=learn_in_RL_flag, distillation_from_teacher_sample_flag=distillation_from_teacher_sample_flag, distillation_update_num=distillation_update_num)

    def update_basic(self, batch_size, expert_value_function=None, episode_num=None):
        batch = self.replay_buffer.sample(batch_size)
        state, action, reward, next_state, done, lstm_info, domain_parameter = batch

        # Updating entropy term
        _, log_prob, std = self.actor.evaluate(state)
        predict_entropy = -log_prob
        entropy_loss = self.entropyTerm.update(predict_entropy.detach())
        self.summaryWriter.add_scalar('status/standard deviation', std.detach().mean().item(), self.summary_writer_count)
        self.summaryWriter.add_scalar('loss/entropy', entropy_loss.detach().item(), self.summary_writer_count)

        # Training Q Function
        new_next_action, next_log_prob, _ = self.actor.evaluate(next_state)
        q1_loss, q2_loss, predicted_q1, predicted_q2 = self.critic.update(state, action, reward, next_state, done,
                                                                          new_next_action.detach(),
                                                                          next_log_prob.detach(), self.entropyTerm.alpha.detach(),
                                                                          domain_parameter,
                                                                          expert_value_function=expert_value_function,
                                                                          episode_num=episode_num)
        self.summaryWriter.add_scalar('status/Q1', predicted_q1.detach().mean().item(), self.summary_writer_count)
        self.summaryWriter.add_scalar('status/Q2', predicted_q2.detach().mean().item(), self.summary_writer_count)
        self.summaryWriter.add_scalar('loss/Q1', q1_loss.detach().item(), self.summary_writer_count)
        self.summaryWriter.add_scalar('loss/Q2', q2_loss.detach().item(), self.summary_writer_count)

        # Training Policy Function
        new_action, log_prob, _ = self.actor.evaluate(state)
        q_value = self.critic.predict_q_value(state, new_action, domain_parameter)
        policy_loss = self.actor.update(self.entropyTerm.alpha.detach(), log_prob, q_value)
        self.summaryWriter.add_scalar('loss/policy', policy_loss.detach().item(), self.summary_writer_count)

        # Q value soft update
        self.critic.soft_update()

        # tensorboard horizonall value
        self.summary_writer_count += 1

    def update_lstm(self, batch_size, expert_value_function=None, episode_num=None):
        batch = self.replay_buffer.sample(batch_size)
        state, action, reward, next_state, done, lstm_term, domain_parameter = batch

        # Updating entropy term
        _, log_prob, std = self.actor.evaluate(state, lstm_term['last_action'], lstm_term['hidden_in'])
        predict_entropy = -log_prob
        entropy_loss = self.entropyTerm.update(predict_entropy.detach())
        self.summaryWriter.add_scalar('status/standard deviation', std.detach().mean().item(), self.summary_writer_count)
        self.summaryWriter.add_scalar('loss/entropy', entropy_loss.detach().item(), self.summary_writer_count)

        # Training Q Function
        new_next_action, next_log_prob, _ = self.actor.evaluate(next_state, action, lstm_term['hidden_out'])
        q1_loss, q2_loss, predicted_q1, predicted_q2 = self.critic.update(state, action, reward, next_state, done,
                                                                          lstm_term, new_next_action.detach(),
                                                                          next_log_prob.detach(), self.entropyTerm.alpha.detach(),
                                                                          domain_parameter,
                                                                          expert_value_function=expert_value_function,
                                                                          episode_num=episode_num)
        self.summaryWriter.add_scalar('status/Q1', predicted_q1.detach().mean().item(), self.summary_writer_count)
        self.summaryWriter.add_scalar('status/Q2', predicted_q2.detach().mean().item(), self.summary_writer_count)
        self.summaryWriter.add_scalar('loss/Q1', q1_loss.detach().item(), self.summary_writer_count)
        self.summaryWriter.add_scalar('loss/Q2', q2_loss.detach().item(), self.summary_writer_count)

        # Training Policy Function
        new_action, log_prob, _ = self.actor.evaluate(state, lstm_term['last_action'], lstm_term['hidden_in'])  # changed
        q_value = self.critic.predict_q_value(state, new_action, lstm_term['last_action'], lstm_term['hidden_in'], domain_parameter)
        policy_loss = self.actor.update(self.entropyTerm.alpha.detach(), log_prob, q_value)
        self.summaryWriter.add_scalar('loss/policy', policy_loss.detach().item(), self.summary_writer_count)

        # Q value soft update
        self.critic.soft_update()

        # tensorboard horizonall value
        self.summary_writer_count += 1

    def DistRL_update_lstm(self, batch_size, expert=None, multi_expert_flag=False, learn_in_RL_flag=True, distillation_from_teacher_sample_flag=False, distillation_update_num=1):
        self.update_RL(learn_in_RL_flag=learn_in_RL_flag, distillation_from_teacher_sample_flag=distillation_from_teacher_sample_flag, expert=expert, batch_size=batch_size)
        self.update_distillation(expert=expert, distillation_update_num=distillation_update_num, distillation_from_teacher_sample_flag=distillation_from_teacher_sample_flag, batch_size=batch_size, multi_expert_flag=multi_expert_flag)
        self.critic.soft_update()
        self.summary_writer_count += 1

    def update_RL(self, learn_in_RL_flag, distillation_from_teacher_sample_flag, expert, batch_size):
        if learn_in_RL_flag is True:
            # make batch
            if distillation_from_teacher_sample_flag is True:
                batch = expert.replay_buffer.sample(batch_size, get_debug_term_flag=True)
            else:
                batch = self.replay_buffer.sample(batch_size, get_debug_term_flag=True)
            state, action, reward, next_state, done, lstm_term, domain_parameter, debug_term = batch

            # Updating entropy term
            _, log_prob, std = self.actor.evaluate(state, lstm_term['last_action'], lstm_term['hidden_in'])
            predict_entropy = -log_prob
            entropy_loss = self.entropyTerm.update(predict_entropy.detach())
            self.summaryWriter.add_scalar('status/standard deviation', std.detach().mean().item(), self.summary_writer_count)
            self.summaryWriter.add_scalar('loss/entropy', entropy_loss.detach().item(), self.summary_writer_count)

            # Training Q Function
            new_next_action, next_log_prob, _ = self.actor.evaluate(next_state, action, lstm_term['hidden_out'])
            q1_loss, q2_loss, predicted_q1, predicted_q2 = self.critic.update(state, action, reward, next_state, done,
                                                                              lstm_term, new_next_action.detach(),
                                                                              next_log_prob.detach(), self.entropyTerm.alpha.detach(),
                                                                              domain_parameter)
            self.summaryWriter.add_scalar('status/Q1', predicted_q1.detach().mean().item(), self.summary_writer_count)
            self.summaryWriter.add_scalar('status/Q2', predicted_q2.detach().mean().item(), self.summary_writer_count)
            self.summaryWriter.add_scalar('loss/Q1', q1_loss.detach().item(), self.summary_writer_count)
            self.summaryWriter.add_scalar('loss/Q2', q2_loss.detach().item(), self.summary_writer_count)

            # Training Policy Function
            new_action, log_prob, _ = self.actor.evaluate(state, lstm_term['last_action'], lstm_term['hidden_in'])  # changed
            q_value = self.critic.predict_q_value(state, new_action, lstm_term['last_action'], lstm_term['hidden_in'], domain_parameter)
            policy_loss = self.actor.update(self.entropyTerm.alpha.detach(), log_prob, q_value)
            self.summaryWriter.add_scalar('loss/policy', policy_loss.detach().item(), self.summary_writer_count)

    def update_distillation(self, batch_size, expert=None, distillation_update_num=1, distillation_from_teacher_sample_flag=False, multi_expert_flag=False):
        if expert is not None:
            """
            alpha=0          : only RL update
            alpha=0~1(const.): distillation x RL
            alpha=0~1(opt.  ): distillation x RL x MI (proposed)
            """
            for update_num in range(distillation_update_num):
                counter = self.summary_writer_count * distillation_update_num + update_num
                # make batch
                if distillation_from_teacher_sample_flag is True:
                    batch = expert.replay_buffer.sample(batch_size, get_debug_term_flag=True)
                else:
                    batch = self.replay_buffer.sample(batch_size, get_debug_term_flag=True)
                state, action, reward, next_state, done, lstm_term, domain_parameter, debug_term = batch

                if self.userDefinedSettings.set_policy_mixture_rate == 0:
                    policy_mixture_rate = self.userDefinedSettings.set_policy_mixture_rate
                else:
                    if multi_expert_flag is True:
                        for one_expert in expert:
                            other_domain_policy_action, _, global_log_std, global_deterministic_action = one_expert.actor.evaluate(state, lstm_term['last_action'], lstm_term['hidden_in'], get_deterministic_action=True)
                            current_domain_policy_action, _, local_log_std, local_deterministic_action = self.actor.evaluate(state, lstm_term['last_action'], lstm_term['hidden_in'], get_deterministic_action=True)
                            self.distillation(counter, state, lstm_term, domain_parameter,
                                              current_domain_action=local_deterministic_action,
                                              support_domain_action=global_deterministic_action)
                    else:
                        other_domain_policy_action, _, global_log_std, global_deterministic_action = expert.actor.evaluate(state, lstm_term['last_action'], lstm_term['hidden_in'], get_deterministic_action=True)
                        current_domain_policy_action, _, local_log_std, local_deterministic_action = self.actor.evaluate(state, lstm_term['last_action'], lstm_term['hidden_in'], get_deterministic_action=True)
                        self.distillation(counter, state, lstm_term, domain_parameter,
                                          current_domain_action=local_deterministic_action,
                                          support_domain_action=global_deterministic_action)

    def distillation(self, counter, state, lstm_term, domain_parameter, current_domain_action=None, support_domain_action=None):
        if self.userDefinedSettings.set_policy_mixture_rate > 0:
            self.calc_policy_mixture_rate(state, current_domain_action, lstm_term, domain_parameter, support_domain_action, counter)
            policy_mixture_rate = self.userDefinedSettings.set_policy_mixture_rate
            target_action = (1. - policy_mixture_rate) * current_domain_action.detach() + policy_mixture_rate * support_domain_action.detach()
            distillation_loss = self.actor.DistRL_update(state, lstm_term['last_action'], lstm_term['hidden_in'], target_action.detach())
            self.summaryWriter.add_scalar('loss/distillation', distillation_loss.detach().item(), counter)
        else:
            policy_mixture_rate = self.calc_policy_mixture_rate(state, current_domain_action, lstm_term, domain_parameter, support_domain_action, counter)
            target_action = (1. - policy_mixture_rate) * current_domain_action.detach() + policy_mixture_rate * support_domain_action.detach()
            distillation_loss = self.actor.DistRL_update(state, lstm_term['last_action'], lstm_term['hidden_in'], target_action.detach())
            self.summaryWriter.add_scalar('loss/distillation', distillation_loss.detach().item(), counter)

    def update_supervised(self, update_num=1, batch_size=None):
        for update_count in range(update_num):
            batch = self.replay_buffer.sample(batch_size=self.userDefinedSettings.batch_size)
            loss = self.actor.direct_update(batch)
            self.summaryWriter.add_scalar('loss/supervised', loss.detach().item(), update_count)

    def calc_policy_mixture_rate(self, state, current_domain_action, lstm_term, domain_parameter, support_domain_action, counter):
        current_domain_q_value = self.critic.predict_q_value(state, current_domain_action.detach(), lstm_term['last_action'], lstm_term['hidden_in'], domain_parameter)
        other_domain_q_value = self.critic.predict_q_value(state, support_domain_action.detach(), lstm_term['last_action'], lstm_term['hidden_in'], domain_parameter)
        advantages = (other_domain_q_value - current_domain_q_value).reshape(-1)
        policy_mixture_rate = torch.clamp(torch.mean(advantages) / torch.max(advantages) + 1., min=0., max=1.)
        self.summaryWriter.add_scalar('status/policy_mixture_rate', policy_mixture_rate.detach().item(), counter)
        return policy_mixture_rate

    def save_model(self, model_dir=None):
        if model_dir is not None:
            self.model_dir = model_dir
        torch.save(self.critic.soft_q_net1.state_dict(), os.path.join(self.model_dir, 'Q1.pth'))
        torch.save(self.critic.soft_q_net2.state_dict(), os.path.join(self.model_dir, 'Q2.pth'))
        torch.save(self.actor.policyNetwork.state_dict(), os.path.join(self.model_dir, 'Policy.pth'))

    def load_model(self, path=None, load_only_policy=False):
        if path is not None:
            self.model_dir = path

        if not load_only_policy:
            self.critic.soft_q_net1.load_state_dict(torch.load(os.path.join(self.model_dir, 'Q1.pth'), map_location=torch.device(self.userDefinedSettings.DEVICE)))
            self.critic.soft_q_net2.load_state_dict(torch.load(os.path.join(self.model_dir, 'Q2.pth'), map_location=torch.device(self.userDefinedSettings.DEVICE)))
            self.critic.soft_q_net1.eval()
            self.critic.soft_q_net2.eval()

        self.actor.policyNetwork.load_state_dict(torch.load(os.path.join(self.model_dir, 'Policy.pth'), map_location=torch.device(self.userDefinedSettings.DEVICE)))
        self.actor.policyNetwork.eval()

    def test(self, model_path=None, target_field=None, domain_num=None, test_num=5, render_flag=True, reward_show_flag=True, deterministic_action_flag=True):
        if model_path is not None:
            self.model_dir = os.path.join(model_path, 'model', str(domain_num))
            self.load_model()

        if target_field is not None:
            actor = target_field.actor
        else:
            actor = self.actor

        total_reward_list = []
        task_achievement_list = []
        for episode_num in range(test_num):
            state = self.env.reset()
            total_reward = 0.

            for step_num in range(self.env.MAX_EPISODE_LENGTH):
                if render_flag is True:
                    self.env.render()

                action, _ = actor.get_action(state, step=step_num, deterministic=deterministic_action_flag)
                next_state, reward, done, domain_parameter, task_achievement = self.env.step(action, get_task_achievement=True)
                state = next_state
                total_reward += reward
                if done:
                    break

            total_reward_list.append(total_reward)
            task_achievement_list.append(task_achievement)

        return np.mean(total_reward_list), sum(task_achievement_list) / len(task_achievement_list)
