# ForgeRL vectorized C++ environment ABI

## Purpose

The ABI isolates high-frequency simulator execution from Python without coupling ForgeRL to a
specific simulator or C++ standard library ABI. Python sees an opaque handle and contiguous NumPy
buffers; an external simulator can implement the same C functions in its own shared library.

The boundary is a C ABI even when the implementation is C++. This avoids exposing STL objects,
exceptions or compiler-specific class layouts across the shared-library boundary.

## Version 1 contract

The header is `cpp/include/forge_rl/vector_env_c_api.h`. A library exports:

```text
forge_rl_counter_env_create      # reference factory; real plugins may add their own factory
forge_rl_vec_env_get_spec
forge_rl_vec_env_reset
forge_rl_vec_env_step
forge_rl_vec_env_last_error
forge_rl_vec_env_destroy
```

`step` receives one discrete action per environment and writes structure-of-arrays outputs:

```text
observations [num_envs, observation_size] float32
rewards      [num_envs]                   float32
terminated   [num_envs]                   uint8
truncated    [num_envs]                   uint8
```

The explicit terminal/truncation split is mandatory. Infrastructure failures are returned as a
non-zero status and must not be converted into a normal terminal transition.

## Ownership and threading

- The creating caller owns the opaque handle and must destroy it exactly once.
- Buffers are caller-owned and must remain valid for the duration of the function call.
- Version 1 does not promise that one handle is thread-safe. Parallelism should be implemented
  inside the vector environment or by using multiple handles.
- Exceptions must not cross the ABI boundary. Implementations convert them to status codes and a
  UTF-8 error string.

## Reference implementation

`cpp/src/counter_vector_env.cpp` is deterministic and asset-free. It is used for ABI, batching and
call-overhead tests. It is not intended to establish reinforcement-learning quality.

A production missile simulator can implement the same interface while keeping proprietary source
and assets outside the public ForgeRL repository.
