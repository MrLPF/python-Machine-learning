#ifndef FORGE_RL_VECTOR_ENV_C_API_H_
#define FORGE_RL_VECTOR_ENV_C_API_H_

#include <stdint.h>

#if defined(_WIN32)
#define FORGE_RL_EXPORT __declspec(dllexport)
#else
#define FORGE_RL_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define FORGE_RL_VEC_ENV_API_VERSION 1

typedef struct forge_rl_vec_env forge_rl_vec_env;

typedef enum forge_rl_action_kind {
  FORGE_RL_ACTION_DISCRETE = 0,
  FORGE_RL_ACTION_CONTINUOUS = 1
} forge_rl_action_kind;

typedef struct forge_rl_vec_env_spec {
  int32_t api_version;
  int32_t num_envs;
  int32_t observation_size;
  int32_t action_size;
  int32_t action_kind;
} forge_rl_vec_env_spec;

typedef enum forge_rl_status {
  FORGE_RL_STATUS_OK = 0,
  FORGE_RL_STATUS_INVALID_ARGUMENT = 1,
  FORGE_RL_STATUS_OUT_OF_RANGE = 2,
  FORGE_RL_STATUS_INTERNAL = 3
} forge_rl_status;

FORGE_RL_EXPORT int32_t forge_rl_counter_env_create(int32_t num_envs,
                                                    forge_rl_vec_env** out_env);
FORGE_RL_EXPORT int32_t forge_rl_vec_env_get_spec(const forge_rl_vec_env* env,
                                                  forge_rl_vec_env_spec* out_spec);
FORGE_RL_EXPORT int32_t forge_rl_vec_env_reset(forge_rl_vec_env* env,
                                               const int32_t* env_ids,
                                               int32_t env_count,
                                               float* observations_out);
FORGE_RL_EXPORT int32_t forge_rl_vec_env_step(forge_rl_vec_env* env,
                                              const int32_t* actions,
                                              float* observations_out,
                                              float* rewards_out,
                                              uint8_t* terminated_out,
                                              uint8_t* truncated_out);
FORGE_RL_EXPORT const char* forge_rl_vec_env_last_error(const forge_rl_vec_env* env);
FORGE_RL_EXPORT void forge_rl_vec_env_destroy(forge_rl_vec_env* env);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // FORGE_RL_VECTOR_ENV_C_API_H_
