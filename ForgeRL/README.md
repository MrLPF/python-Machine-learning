# ForgeRL

ForgeRL is a high-throughput distributed reinforcement-learning runtime built on PyTorch. PyTorch
continues to own tensor kernels, autograd, CUDA allocation, mixed precision, DDP and future FSDP2
support. ForgeRL owns the reinforcement-learning dataflow:

- vectorized environment execution and fault isolation;
- node-local, deadline-aware inference batching;
- versioned and atomically activated policies;
- trajectory/replay consistency and backpressure;
- multi-process and multi-node learner orchestration;
- PPO/APPO/V-trace first, with extension points for MAPPO, SAC and R2D2.

The implementation is being migrated incrementally from the uploaded ForgeRL v1 design. It is
intentionally **not** a replacement for PyTorch or OneFlow's tensor/autograd layer.

## Implemented foundations

### M0 â€” correctness and distributed foundation

- clean installation without simulator-specific dependencies;
- explicit `terminated`, `truncated` and invalid-system-transition contracts;
- validated `TransitionBatch` with unique step identity;
- bounded on-policy experience queue with policy-lag and age filtering;
- versioned policy registry and coordinator lease/heartbeat contracts;
- PyTorch DDP learner utilities and two-process Gloo CI.

### M1 â€” local high-throughput runtime and acceptance tooling

- generation-safe shared-memory tensor-tree channels;
- compact slot descriptors instead of pickled tensor payloads;
- cross-actor deadline/minimum-size dynamic inference batching;
- one fair collector thread for all node-local actor endpoints;
- bounded per-endpoint draining to prevent a hot actor from starving peers;
- zero-copy one-request batches and reusable multi-request assembly buffers;
- active/staging double-buffered policy replicas;
- optional pinned-memory, non-blocking H2D and AMP inference path;
- lossless trajectory fragments with separate bootstrap state;
- stable C ABI for vectorized C++ environments and a deterministic reference environment;
- source-fingerprinted v1 Queue/pickle predictor reference;
- same-model v1/v2 synthetic transition audit;
- shared reference PPO core for CartPole/Pendulum learning-quality comparison;
- normal-CI learning plumbing smoke and a pinned self-hosted five-seed workflow.

The previous per-actor-thread implementation is retained as
`ThreadedNodeLocalInferenceService` for regression comparison. The default
`NodeLocalInferenceService` reports collector, allocation, copied-byte, reused-buffer and
single-request zero-copy counters.

M1 is **not performance accepted**. Acceptance still requires a pinned-hardware `>=2x` valid-row
result and a formal five-seed report for both environments whose final `m1-final.json` stattÈ\Â˜ÓØˆÜ™Y[ˆÜÝYÒH[Û™H\È›Ý]]šY[˜ÙK‚‚ˆÈÈ]ZXÚÈÝ\‚˜˜\Úœ]Ûˆ[H™[ˆ™[‚œÛÝ\˜ÙH™[‹Øš[‹ØXÝ]˜]Bœ\[œÝ[YH	Ë–Ù]‹Þ[WIÂœ]\Ý\Bœ]ÛˆØÜš\ËØ™[˜ÚX\š×Ù[‹œHKY[ˆØ\ÛK]ŒHK\Ý\ÈLœ]ÛˆØÜš\ËØ™[˜ÚX\š×Ü[[YKœHKZ][\ÈŒœ]ÛˆØÜš\ËØ™[˜ÚX\š×ÛLWÚ[™™\™[˜ÙKœHˆKXXÝÜœÈK\™\]Y\ÝË\\‹XXÝÜˆLˆKXÛÛXÝÜ‹\Û[\ÈŒHK[X^Y˜Z[‹\\‹Y[™Ú[œ]ÛˆØÜš\ËØ™[˜ÚX\š×ÛLWØXØÙ\[˜ÙKœHˆKXXÝÜœÈˆK\™\]Y\ÝË\\‹XXÝÜˆKZ][\Ë\\‹\™\]Y\ÝˆˆK]ÚYMˆK[X^X˜]ÚZ][\ÈK]Œ‹[Z[‹X˜]ÚZ][\ÈˆK]›ÝYÚ]YØ]Hœ]ÛˆØÜš\ËØ™[˜ÚX\š×ÛLWÛX\›š[™ËœHˆK\Û[ÚÙHK\ÙYYÈÈˆKY[š\›Û›Y[ÈØ\ÛK]ŒK[™[[K]ŒHˆK[Ý]]™[˜ÚX\šÜËÜ™\Ý[ËÛLK[X\›š[™Ë\Û[ÚÙKšœÛÛ‚˜‚ˆÈÈ›Ü›X[LH^XÝ][Û‚‚•HX[X[™Ú]X‹ÝÛÜšÙ›ÝÜËÙ›Ü™ÙK\›[LKY›Ü›X[ž[[ÛÜšÙ›ÝÈ\™Ù]ÈH[›™YÙ[‹ZÜÝY[›™\‚›X™[Y›Ü™Ù\›X™[˜ÚX\šØˆ][œÈH›Ü›X[Z\™Yš]™K\ÙYYX\›š[™ÈØ]KHÛÛ›ÛYœÞ[]XÈ›ÝYÚ]Ø]K[™Hš[˜[K\™\]Z\™KYÛØXÚ\Ú[ÛˆÚ]Ý]™\XÚ[™ÈH[›™\‰ÜÂœ™K\›Ýš\Ú[Û™Y]Ûˆ[š\›Û›Y[‚‚ˆÈÈ[š]X[™[˜ÚX\šÈ[š\›Û›Y[Â‚‹HØ\ÛK]ŒXˆ\ØÜ™]KXXÝ[ÛˆÛÜœ™XÝ™\ÜÈ[™[YK]Ë]\™Ù]Â‹H[™[[K]ŒXˆÛÛ[[Ý\ËXXÝ[ÛˆÛÜœ™XÝ™\ÜÈ[™[YK]Ë]\™Ù]Â‹H][™Ö›ÛÈTHÚ[\WÜÜ™XYÝŒØˆ]\ˆ][KXYÙ[ÓPTÈ\Ý[™ÎÂ‹HÜ[Û˜[Þ[[˜\Ú][H]R›ÐÛÈ[]Xˆ]\ˆÛÝYØØ[[™È[™ÛÛ[[Ý\ÈÛÛ›Û‚‚•H™Y™\™[˜ÙHÈ\È[ˆLHXØÙ\[˜ÙHÝXš™XÝ›ÝÛÛ\][ÛˆÙˆHLÈ[ÛÜš]K\YÚ[ˆÛÜšËˆBÊÊÈÛÝ[\ˆ[š\›Û›Y[\ÈH[[YKÐP’H™[˜ÚX\šË›ÝHX\›š[™Ë\]X[]H™[˜ÚX\šË‚‚ˆÈÈ™\ÜÚ]ÜžHÝ]\Â‚XØÙ\[˜ÙHØ]\È[™™[XZ[š[™ÈÛÜšÈ\™H˜XÚÙY[ˆØØÜËÔ“ÐQPT›YJØÜËÔ“ÐQPT›Y
KˆB˜™[˜ÚX\šÈY]ÙÛÙÞH\ÈYš[™Y[ˆØØÜËÐ‘SÒPT’ÔË›YJØÜËÐ‘SÒPT’ÔË›Y
K[™H^XÝLBœ›ØÙY\™H\È[ˆØØÜËÓLWÐPÐÑTSÑK›YJØÜËÓLWÐPÐÑTSÑK›Y
KˆHÊÊÈ[š\›Û›Y[›Ý[™\žH\Â™ØÝ[Y[Y[ˆØØÜËÐÔÕ‘PÕÔ—ÑS•—ÐP’K›YJØÜËÐÔÕ‘PÕÔ—ÑS•—ÐP’K›Y
K‚