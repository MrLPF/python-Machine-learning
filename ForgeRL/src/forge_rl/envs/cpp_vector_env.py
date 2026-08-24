from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


class _CVecEnvSpec(ctypes.Structure):
    _fields_ = [
        ("api_version", ctypes.c_int32),
        ("num_envs", ctypes.c_int32),
        ("observation_size", ctypes.c_int32),
        ("action_size", ctypes.c_int32),
        ("action_kind", ctypes.c_int32),
    ]


@dataclass(frozen=True, slots=True)
class VectorEnvSpec:
    api_version: int
    num_envs: int
    observation_size: int
    action_size: int
    action_kind: str


@dataclass(frozen=True, slots=True)
class VectorStep:
    observations: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray


class CppVectorEnv:
    """ctypes wrapper for the stable ForgeRL vector-environment C ABI."""

    API_VERSION = 1

    def __init__(self, library_path: str | Path, *, num_envs: int) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.library_path = Path(library_path)
        self._library = ctypes.CDLL(str(self.library_path))
        self._configure_api()
        self._handle = ctypes.c_void_p()
        status = self._library.forge_rl_counter_env_create(
            ctypes.c_int32(num_envs), ctypes.byref(self._handle)
        )
        self._check(status)
        self._closed = False
        self.spec = self._get_spec()
        if self.spec.api_version != self.API_VERSION:
            self.close()
            raise RuntimeError(
                f"C++ vector-env ABI {self.spec.api_version} != supported {self.API_VERSION}"
            )

    def _configure_api(self) -> None:
        library = self._library
        library.forge_rl_counter_env_create.argtypes = [
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        library.forge_rl_counter_env_create.restype = ctypes.c_int32
        library.forge_rl_vec_env_get_spec.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_CVecEnvSpec),
        ]
        library.forge_rl_vec_env_get_spec.restype = ctypes.c_int32
        library.forge_rl_vec_env_reset.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_float),
        ]
        library.forge_rl_vec_env_reset.restype = ctypes.c_int32
        library.forge_rl_vec_env_step.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_uint8),
        ]
        library.forge_rl_vec_env_step.restype = ctypes.c_int32
        library.forge_rl_vec_env_last_error.argtypes = [ctypes.c_void_p]
        library.forge_rl_vec_env_last_error.restype = ctypes.c_char_p
        library.forge_rl_vec_env_destroy.argtypes = [ctypes.c_void_p]
        library.forge_rl_vec_env_destroy.restype = None

    def _check(self, status: int) -> None:
        if int(status) == 0:
            return
        message = self._library.forge_rl_vec_env_last_error(self._handle)
        detail = message.decode("utf-8", errors="replace") if message else "unknown C++ error"
        raise RuntimeError(f"ForgeRL vector-env status {status}: {detail}")

    def _get_spec(self) -> VectorEnvSpec:
        raw = _CVecEnvSpec()
        self._check(self._library.forge_rl_vec_env_get_spec(self._handle, ctypes.byref(raw)))
        kinds = {0: "discrete", 1: "continuous"}
        if raw.action_kind not in kinds:
            raise RuntimeError(f"unknown action kind {raw.action_kind}")
        return VectorEnvSpec(
            api_version=int(raw.api_version),
            num_envs=int(raw.num_envs),
            observation_size=int(raw.observation_size),
            action_size=int(raw.action_size),
            action_kind=kinds[int(raw.action_kind)],
        )

    def reset(self, env_ids: np.ndarray | list[int] | None = None) -> np.ndarray:
        self._ensure_open()
        selected = (
            np.arange(self.spec.num_envs, dtype=np.int32)
            if env_ids is None
            else np.ascontiguousarray(env_ids, dtype=np.int32)
        )
        if selected.ndim != 1 or selected.size == 0:
            raise ValueError("env_ids must be a non-empty one-dimensional array")
        observations = np.empty(
            (selected.size, self.spec.observation_size), dtype=np.float32
        )
        self._check(
            self._library.forge_rl_vec_env_reset(
                self._handle,
                selected.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                ctypes.c_int32(selected.size),
                observations.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            )
        )
        return observations

    def step(self, actions: np.ndarray | list[int]) -> VectorStep:
        self._ensure_open()
        selected = np.ascontiguousarray(actions, dtype=np.int32)
        if selected.shape != (self.spec.num_envs,):
            raise ValueError(f"actions must have shape ({self.spec.num_envs},)")
        observations = np.empty(
            (self.spec.num_envs, self.spec.observation_size), dtype=np.float32
        )
        rewards = np.empty(self.spec.num_envs, dtype=np.float32)
        terminated = np.empty(self.spec.num_envs, dtype=np.uint8)
        truncated = np.empty(self.spec.num_envs, dtype=np.uint8)
        self._check(
            self._library.forge_rl_vec_env_step(
                self._handle,
                selected.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                observations.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                rewards.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                terminated.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                truncated.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            )
        )
        return VectorStep(
            observations=observations,
            rewards=rewards,
            terminated=terminated.astype(np.bool_),
            truncated=truncated.astype(np.bool_),
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("C++ vector environment is closed")

    def close(self) -> None:
        if getattr(self, "_closed", True):
            return
        self._library.forge_rl_vec_env_destroy(self._handle)
        self._handle = ctypes.c_void_p()
        self._closed = True

    def __enter__(self) -> "CppVectorEnv":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except BaseException:
            pass
