from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from forge_rl.envs import CppVectorEnv


def test_cpp_counter_vector_env_c_abi(tmp_path: Path) -> None:
    compiler = shutil.which("g++") or shutil.which("c++")
    if compiler is None:
        pytest.skip("a C++ compiler is required for the vector-env ABI test")
    project = Path(__file__).resolve().parents[1]
    library = tmp_path / "libforge_rl_counter_env.so"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O3",
            "-shared",
            "-fPIC",
            "-I",
            str(project / "cpp" / "include"),
            str(project / "cpp" / "src" / "counter_vector_env.cpp"),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    with CppVectorEnv(library, num_envs=4) as env:
        assert env.spec.api_version == 1
        assert env.spec.action_kind == "discrete"
        reset = env.reset()
        np.testing.assert_array_equal(reset, np.zeros((4, 2), dtype=np.float32))
        step = env.step([1, 1, 0, 0])
        np.testing.assert_allclose(
            step.observations[:, 0],
            np.asarray([0.1, 0.1, -0.1, -0.1], dtype=np.float32),
        )
        np.testing.assert_array_equal(step.rewards, -np.ones(4, dtype=np.float32))
        assert not step.terminated.any()
        assert not step.truncated.any()
        partial = env.reset([1, 3])
        np.testing.assert_array_equal(partial, np.zeros((2, 2), dtype=np.float32))
