import numpy as np
import torch


class PendulumDomainInfo(object):
    def __init__(self, userDefinedSettings, domain_range=None):
        self.userDefinedSettings = userDefinedSettings
        self.DOMAIN_RANDOMIZATION_FLAG = userDefinedSettings.DOMAIN_RANDOMIZATION_FLAG
        self.set_domain_parameter_all_space(domain_range=domain_range, randomization_flag=userDefinedSettings.DOMAIN_RANDOMIZATION_FLAG)

    def set_domain_parameter_all_space(self, domain_range=None, randomization_flag=True):
        if randomization_flag:
            sampling_method = 'uniform'
        else:
            sampling_method = 'fix'
        self.dt = DomainParameter(name='dt', initial_value=0.05, min_value=0.04, max_value=0.06, sampling_method=sampling_method)
        self.g = DomainParameter(name='g', initial_value=10., min_value=7, max_value=15., sampling_method=sampling_method)
        self.m = DomainParameter(name='m', initial_value=1., min_value=0.8, max_value=1.2, sampling_method=sampling_method)
        self.l = DomainParameter(name='l', initial_value=1., min_value=0.8, max_value=1.2, sampling_method=sampling_method)
        self.torque_weight = DomainParameter(name='torque_weight', initial_value=1., min_value=0.7, max_value=1.5, sampling_method=sampling_method)
        self.torque_bias = DomainParameter(name='torque_bias', initial_value=0, min_value=-0.5, max_value=0.5, sampling_method=sampling_method)
        if self.DOMAIN_RANDOMIZATION_FLAG:
            self.torque_limit = False
            self.velocity_limit = False
        else:
            self.torque_limit = True
            self.velocity_limit = True

    def set_parameters(self, set_info=None, type=None):
        if self.DOMAIN_RANDOMIZATION_FLAG:
            if type == 'set_split2':
                target_domains = [self.g, self.torque_weight]
                other_domains = [self.dt, self.m, self.l, self.torque_bias]
                for domain, set_value in zip(target_domains, set_info):
                    domain.set(set_value=set_value, set_method='rate_set')
                for domain in other_domains:
                    domain.set()
            elif set_info is not None:
                for parameter, set_range in zip([self.dt, self.g, self.m, self.l, self.torque_weight, self.torque_bias], set_info):
                    parameter.set(set_range=set_range)
            else:
                for parameter in [self.dt, self.g, self.m, self.l, self.torque_weight, self.torque_bias]:
                    parameter.set()

    def get_domain_parameters(self, normalize_flag=False):
        domain_parameters = []
        for parameter in [self.dt, self.g, self.m, self.l, self.torque_weight, self.torque_bias]:
            if normalize_flag:
                value = (parameter.value - parameter.min_value) / (parameter.max_value - parameter.min_value)  # (a,b)->(0,1)
                domain_parameters.append(value)
            else:
                domain_parameters.append(parameter.value)
        domain_parameters = np.array(domain_parameters)
        return domain_parameters

    def normalize(self, domain_parameters):
        min_value = []
        max_value = []
        for parameter in [self.dt, self.g, self.m, self.l, self.torque_weight, self.torque_bias]:
            min_value.append(parameter.min_value)
            max_value.append(parameter.max_value)
        min_value = torch.Tensor(min_value).to(self.userDefinedSettings.DEVICE)
        max_value = torch.Tensor(max_value).to(self.userDefinedSettings.DEVICE)
        normalized_domain_parameters = (domain_parameters - min_value) / (max_value - min_value)  # (a,b)->(0,1)
        return normalized_domain_parameters


class DomainParameter(object):
    def __init__(self, name, initial_value, min_value, max_value, sampling_method):
        assert (initial_value >= min_value) and (initial_value <= max_value), 'domain initial value is out of range'
        self.name = name
        self.value = initial_value
        self.min_value = min_value
        self.max_value = max_value
        self.min_range = 0.
        self.max_range = 1.
        self.sampling_method = sampling_method

    def set(self, set_value=None, set_range=None, set_method=None):
        if set_method == 'direct_set':
            self.value = set_value
        elif set_method == 'rate_set':
            # range [0,1]
            self.value = (self.max_value - self.min_value) * set_value + self.min_value
        else:
            if self.sampling_method == 'uniform':
                if set_range is not None:
                    self.set_divided_space(set_range)
                origin_sample = np.random.rand()
                shifted_sample = (self.max_range - self.min_range) * origin_sample + self.min_range
                self.value = (self.max_value - self.min_value) * shifted_sample + self.min_value
            elif self.sampling_method == 'fix':
                pass
            elif self.sampling_method == 'set':
                self.value = set_value
            else:
                assert False, 'choose sampling method of the domain parameter'

        assert (self.value >= self.min_value) and (self.value <= self.max_value), 'domain value is out of range: {} < {} < {}'.format(self.min_value, self.value, self.max_value)

    def set_divided_space(self, domain_range):
        self.min_range = domain_range['min']
        self.max_range = domain_range['max']
