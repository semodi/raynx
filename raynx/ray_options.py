from typing import Any, Union

from pydantic import BaseModel


class RayRemoteOptions(BaseModel):
    """(Resource) options to pass to ray.remote.
    (From ray documentation:)
    Args:
        use_ray (bool): Whether to use ray. Defaults to True.
        num_returns: This is only for *remote functions*. It specifies
            the number of object refs returned by the remote function
            invocation. Pass "dynamic" to allow the task to decide how many
            return values to return during execution, and the caller will
            receive an ObjectRef[ObjectRefGenerator] (note, this setting is
            experimental).
        num_cpus: The quantity of CPU cores to reserve
            for this task or for the lifetime of the actor.
        num_gpus: The quantity of GPUs to reserve
            for this task or for the lifetime of the actor.
        resources (Dict[str, float]): The quantity of various custom resources
            to reserve for this task or for the lifetime of the actor.
            This is a dictionary mapping strings (resource names) to floats.
        accelerator_type: If specified, requires that the task or actor run
            on a node with the specified type of accelerator.
            See `ray.accelerators` for accelerator types.
        memory: The heap memory request for this task/actor.
        max_calls: Only for *remote functions*. This specifies the
            maximum number of times that a given worker can execute
            the given remote function before it must exit
            (this can be used to address memory leaks in third-party
            libraries or to reclaim resources that cannot easily be
            released, e.g., GPU memory that was acquired by TensorFlow).
            By default this is infinite.
        max_restarts: Only for *actors*. This specifies the maximum
            number of times that the actor should be restarted when it dies
            unexpectedly. The minimum valid value is 0 (default),
            which indicates that the actor doesn't need to be restarted.
            A value of -1 indicates that an actor should be restarted
            indefinitely.
        max_task_retries: Only for *actors*. How many times to
            retry an actor task if the task fails due to a system error,
            e.g., the actor has died. If set to -1, the system will
            retry the failed task until the task succeeds, or the actor
            has reached its max_restarts limit. If set to `n > 0`, the
            system will retry the failed task up to n times, after which the
            task will throw a `RayActorError` exception upon :obj:`ray.get`.
            Note that Python exceptions are not considered system errors
            and will not trigger retries.
        max_retries: Only for *remote functions*. This specifies
            the maximum number of times that the remote function
            should be rerun when the worker process executing it
            crashes unexpectedly. The minimum valid value is 0,
            the default is 4 (default), and a value of -1 indicates
            infinite retries.
        runtime_env (Dict[str, Any]): Specifies the runtime environment for
            this actor or task and its children. See
            :ref:`runtime-environments` for detailed documentation. This API is
            in beta and may change before becoming stable.
        retry_exceptions: Only for *remote functions*. This specifies whether
            application-level errors should be retried up to max_retries times.
            This can be a boolean or a list of exceptions that should be retried.
        scheduling_strategy: Strategy about how to
            schedule a remote function or actor. Possible values are
            None: ray will figure out the scheduling strategy to use, it
            will either be the PlacementGroupSchedulingStrategy using parent's
            placement group if parent has one and has
            placement_group_capture_child_tasks set to true,
            or "DEFAULT";
            "DEFAULT": default hybrid scheduling;
            "SPREAD": best effort spread scheduling;
            `PlacementGroupSchedulingStrategy`:
            placement group based scheduling.
    """

    use_ray: bool = False
    num_returns: Union[int, float] = None
    num_cpus: Union[int, float] = None
    num_gpus: Union[int, float] = None
    resources: dict[str, float] = None
    accelerator_type: str = None
    memory: Union[int, float] = None
    max_calls: int = None
    max_restarts: int = None
    max_task_retries: int = None
    max_retries: int = None
    runtime_env: dict[str, Any] = None
    retry_exceptions: bool = None
    scheduling_strategy: str = None

    @property
    def set_remote_options(self):
        """Only return options (fields) that have been set explicitly"""
        return {
            key: val
            for key, val in self.dict(exclude={"use_ray": True}).items()
            if val is not None
        }




class RayGlob:
    def __init__(self, remote_options: RayRemoteOptions = None):
        """Ray global options

        Args:
            remote_options (RayRemoteOptions, optional): ray remote options.
                 Defaults to RayRemoteOptions().
        """
        self.reinit(remote_options or RayRemoteOptions())

    def reinit(self, remote_options: RayRemoteOptions):
        self._remote_options = remote_options or RayRemoteOptions()
        self._use_ray = remote_options.use_ray

    def ray_on(self):
        """Turn ray on"""
        self._use_ray = True

    def ray_off(self):
        """Turn ray off"""
        self._use_ray = False

    @property
    def remote_options(self) -> dict:
        return self._remote_options.set_remote_options

    @property
    def use_ray(self):
        return self._use_ray


RayGlob = RayGlob()
