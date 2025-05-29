import abc


class Module(abc.ABC):
    @abc.abstractmethod
    def init(self, key):
        raise NotImplementedError

    @abc.abstractmethod
    def apply(self, param, *args, **kwargs):
        raise NotImplementedError

    def parameter_loss(self, param):
        return 0.0
