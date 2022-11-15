import functools
from copy import deepcopy
from typing import Callable

import ray
from raynx.ray_options import RayGlob



class RayFuture:
    """A safe future, which implements ``get`` that retrieves
    results regardless of whether ray was used or not"""

    def __init__(self, val):
        self.val = val

    def get(self):
        try:
            return ray.get(self.val)
        except ValueError:
            return self.val


def method_to_function(method: Callable):
    """Converts a method into a function to by-pass ray remote
    checks"""

    @functools.wraps(method)
    def wrapper(*args, **kwargs):
        return method(*args, **kwargs)

    return wrapper


class RaySwitchRemote:
    def __init__(self, func, context = None):
        """RaySwitchRemote wraps a function to provide the .remote() method.
        .remote() will have the expected behavior if ray is activated and will
        simply return a mock future if ray is deactivated. The resulting future can
        be resolved to its return value with future.get(), which, in the case ray
        is being used, is a blocking function."""
        try:
            self._func_remote = ray.remote(func)
        except TypeError:
            self._func_remote = ray.remote(method_to_function(func))

        self.context = context
        self._func = func
        self._options = {}

    @property
    def ray(self):
        return (
            self.context.ray.use_ray if self.context and self.context.ray else False
        ) or RayGlob.use_ray

    def _copy_with_options(self, **options):
        rs = RaySwitchRemote(self._func, context=self.context)
        rs._options = options
        return rs

    def options(self, **kwargs):
        return self._copy_with_options(**kwargs)

    def remote(self, *args, **kwargs) -> RayFuture:
        remote_options = deepcopy(RayGlob.remote_options)
        if self.context and self.context.ray:
            remote_options.update(self.context.ray.set_remote_options)
        remote_options.update(self._options)
        if self.ray:
            if remote_options:
                callable = self._func_remote.options(**remote_options)
            else:
                callable = self._func_remote
            return RayFuture(callable.remote(*args, **kwargs))
        else:
            return RayFuture(self._func(*args, **kwargs))
