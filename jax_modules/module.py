import abc


class Module(abc.ABC):
    @abc.abstractmethod
    def init(self, key):
        raise NotImplementedError

    @abc.abstractmethod
    def apply(self, params, *args, **kwargs):
        raise NotImplementedError

    def param_loss(self, param):
        return 0.0
