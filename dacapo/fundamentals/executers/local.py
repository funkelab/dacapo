from .helpers import Executer


class Local(Executer):
    def train(self):
        raise NotImplementedError()

    def validate(self):
        raise NotImplementedError()

    def apply(self):
        raise NotImplementedError()

    def post_process(self):
        raise NotImplementedError()
