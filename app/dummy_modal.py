# dummy_modal.py
class Stub:
    def __init__(self, *args, **kwargs):
        pass

    def cls(self, *args, **kwargs):
        def decorator(cls):
            return cls
        return decorator

    def local_entrypoint(self, *args, **kwargs):
        def decorator(func):
            return func
        return decorator

    def function(self, *args, **kwargs):
        def decorator(func):
            return func
        return decorator

class Secret:
    @staticmethod
    def from_name(*args, **kwargs):
        pass

class Mount:
    @staticmethod
    def from_local_python_packages(*args, **kwargs):
        pass

class Image:
    @staticmethod
    def from_registry(*args, **kwargs):
        return Image()

    def pip_install(self, *args, **kwargs):
        return self

    def run_commands(self, *args, **kwargs):
        return self

def asgi_app(*args, **kwargs):
    def decorator(func):
        return func
    return decorator

def method(*args, **kwargs):
    def decorator(func):
        return func
    return decorator