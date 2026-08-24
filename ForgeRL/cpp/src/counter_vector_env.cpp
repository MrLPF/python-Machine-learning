#include "forge_rl/vector_env_c_api.h"

#include <cmath>
#include <exception>
#include <new>
#include <string>
#include <vector>

struct forge_rl_vec_env {
  explicit forge_rl_vec_env(int32_t count)
      : num_envs(count), position(count, 0), steps(count, 0) {}

  int32_t num_envs;
  std::vector<int32_t> position;
  std::vector<int32_t> steps;
  std::string last_error;
};

namespace {

thread_local std::string g_last_error;

int32_t Fail(forge_rl_vec_env* env, int32_t status, const char* message) {
  if (env != nullptr) {
    env->last_error = message;
  } else {
    g_last_error = message;
  }
  return status;
}

void WriteObservation(const forge_rl_vec_env& env, int32_t env_id, float* output) {
  output[0] = static_cast<float>(env.position[env_id]) / 10.0F;
  output[1] = static_cast<float>(env.steps[env_id]) / 100.0F;
}

}  // namespace

extern "C" {

int32_t forge_rl_counter_env_create(int32_t num_envs, forge_rl_vec_env** out_env) {
  if (out_env == nullptr || num_envs <= 0) {
    return Fail(nullptr, FORGE_RL_STATUS_INVALID_ARGUMENT,
                "num_envs must be positive and out_env must be non-null");
  }
  try {
    *out_env = new forge_rl_vec_env(num_envs);
    return FORGE_RL_STATUS_OK;
  } catch (const std::exception& error) {
    return Fail(nullptr, FORGE_RL_STATUS_INTERNAL, error.what());
  }
}

int32_t forge_rl_vec_env_get_spec(const forge_rl_vec_env* env,
                                  forge_rl_vec_env_spec* out_spec) {
  if (env == nullptr || out_spec == nullptr) {
    return Fail(const_cast<forge_rl_vec_env*>(env), FORGE_RL_STATUS_INVALID_ARGUMENT,
                "env and out_spec must be non-null");
  }
  out_spec->api_version = FORGE_RL_VEC_ENV_API_VERSION;
  out_spec->num_envs = env->num_envs;
  out_spec->observation_size = 2;
  out_spec->action_size = 2;
  out_spec->action_kind = FORGE_RL_ACTION_DISCRETE;
  return FORGE_RL_STATUS_OK;
}

int32_t forge_rl_vec_env_reset(forge_rl_vec_env* env, const int32_t* env_ids,
                               int32_t env_count, float* observations_out) {
  if (env == nullptr || env_ids == nullptr || observations_out == nullptr ||
      env_count <= 0) {
    return Fail(env, FORGE_RL_STATUS_INVALID_ARGUMENT,
                "env, env_ids and observations_out must be non-null and env_count positive");
  }
  for (int32_t index = 0; index < env_count; ++index) {
    const int32_t env_id = env_ids[index];
    if (env_id < 0 || env_id >= env->num_envs) {
      return Fail(env, FORGE_RL_STATUS_OUT_OF_RANGE, "reset env_id is out of range");
    }
    env->position[env_id] = 0;
    env->steps[env_id] = 0;
    WriteObservation(*env, env_id, observations_out + index * 2);
  }
  return FORGE_RL_STATUS_OK;
}

int32_t forge_rl_vec_env_step(forge_rl_vec_env* env, const int32_t* actions,
                              float* observations_out, float* rewards_out,
                              uint8_t* terminated_out, uint8_t* truncated_out) {
  if (env == nullptr || actions == nullptr || observations_out == nullptr ||
      rewards_out == nullptr || terminated_out == nullptr || truncated_out == nullptr) {
    return Fail(env, FORGE_RL_STATUS_INVALID_ARGUMENT, "step pointers must be non-null");
  }
  for (int32_t env_id = 0; env_id < env->num_envs; ++env_id) {
    const int32_t action = actions[env_id];
    if (action != 0 && action != 1) {
      return Fail(env, FORGE_RL_STATUS_OUT_OF_RANGE, "counter action must be 0 or 1");
    }
    env->position[env_id] += action == 0 ? -1 : 1;
    env->steps[env_id] += 1;
    const bool terminated = std::abs(env->position[env_id]) >= 10;
    const bool truncated = !terminated && env->steps[env_id] >= 100;
    rewards_out[env_id] = -static_cast<float>(std::abs(env->position[env_id]));
    terminated_out[env_id] = terminated ? 1 : 0;
    truncated_out[env_id] = truncated ? 1 : 0;
    WriteObservation(*env, env_id, observations_out + env_id * 2);
  }
  return FORGE_RL_STATUS_OK;
}

const char* forge_rl_vec_env_last_error(const forge_rl_vec_env* env) {
  if (env == nullptr) {
    return g_last_error.c_str();
  }
  return env->last_error.c_str();
}

void forge_rl_vec_env_destroy(forge_rl_vec_env* env) { delete env; }

}  // extern "C"
