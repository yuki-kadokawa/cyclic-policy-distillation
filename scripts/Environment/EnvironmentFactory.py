
class EnvironmentFactory():
    def __init__(self, userDefinedSettings):
        self.ENVIRONMENT_NAME = userDefinedSettings.ENVIRONMENT_NAME
        self.userDefinedSettings = userDefinedSettings

    def generate(self, domain_range=None):

        if self.ENVIRONMENT_NAME == 'Pendulum':
            from .Pendulum.Pendulum import Pendulum
            return Pendulum(self.userDefinedSettings, domain_range=domain_range)
