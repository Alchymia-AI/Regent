# DSTP — Distributed Shared Training Protocol

DSTP executes language model training runs across a pool of untrusted, geographically distributed GPU contributors over the public internet. Initiators publish signed **training plans**; contributors subscribe to plans matching their hardware and preferences; multiple plans run concurrently on the same pool.

DSTP does not choose the algorithm, the model, or the data. Those are plan parameters.

---

## 1. Scope

In:
- Work distribution across unreliable contributors
- Cryptographic verification of submitted work
- Coordinator consensus with churn
- Reproducible checkpoints from the shard ledger

Out:
- Algorithm, architecture, data curation (plan parameters)
- Payment custody (off-protocol)
- Inference

Required but not provided:
- A low-communication outer-loop algorithm. WAN training is infeasible without one. DiLoCo is the reference.

---

## 2. Terms

| Term | Definition |
|---|---|
| Registry | Global directory of runs, contributors, reputations. Holds no weights, no funds. |
| Run | One training job. Defined by a signed plan. Has its own coordinator and ledger. |
| Initiator | Publishes the plan. Operates the coordinator. |
| Coordinator | Raft-replicated state machine. Holds global weights and shard ledger. |
| Contributor | Runs the reference runtime. Executes inner-loop training. |
| Shard | `(data_slice, starting_weights, seed, inner_steps)` → `(delta, ending_hash)`. |
| Verifier | Re-executes sampled shards. Declared in the plan. |
| Plan | Signed YAML manifest. Immutable once published. See §4. |

All agents hold Ed25519 keys. All messages are signed. All artifacts are content-addressed.

---

## 3. Invariants

1. Global weights at round `N` are a deterministic function of `(plan, seed_checkpoint, accepted_shards[0..N])`.
2. A shard's `ending_hash` is reproducible from its inputs under the plan's runtime.
3. Acceptance requires either `peer_k` matching submissions or verifier confirmation.
4. State transitions gate on verification. Rejected work cannot reach the ledger.
5. The coordinator reconstructs from the shard ledger alone.

---

## 4. Training Plan

```yaml
run_id:        uuid
version:       1
initiator:     { pubkey, contact }

model:
  architecture: string            # e.g., "regent-mamba2"
  config_cid:   cid
  seed_cid:     cid | null

data:
  corpus_cid:   cid
  tokenizer_cid: cid
  total_bytes:  int

algorithm:
  outer:         "diloco" | "fedavg" | "custom"
  inner:         "adamw" | ...
  runtime_cid:   cid                # container image

hyperparameters:
  inner_steps:   int
  inner_batch:   int
  seq_len:       int
  inner_lr:      float
  outer_lr:      float
  dtype:         "fp16" | "bf16" | "fp32"
  seed_base:     int

shard:
  shards_total:  int
  shard_bytes:   int
  capacity_tiers:
    - { vram_gb: 80, batch_mul: 1.0 }
    - { vram_gb: 40, batch_mul: 0.5 }
    - { vram_gb: 24, batch_mul: 0.25 }

verification:
  peer_k:        int
  spot_rate:     float              # fraction re-executed
  verifiers:     [pubkey]
  hash_mode:     "exact" | "approx"

compression:
  gradient_bits: 8
  topk:          0.1
  error_feedback: true

coordinator:
  replicas:      [endpoint]
  pubkey:        ed25519

incentive:
  model:            "credit" | "fiat" | "token" | "hybrid"
  reward_per_shard: decimal | null
  currency:         string | null
  schedule:         string

deadline:        iso8601 | null
signature:       sig
```

---

## 5. Protocol

### Messages

Protobuf over gRPC with mutual TLS. All signed.

```
AnnounceRun(plan)                       → ack
ListRuns(filter)                        → [run_summary]
GetPlan(run_id)                         → plan
RegisterContributor(pubkey, spec)       → ack

Join(run_id, contributor_id)            → ack
RequestShard(contributor_id, tier)      → shard_assignment
SubmitDelta(shard_id, delta_cid, ending_hash, sig)
                                        → ack | reject(reason)
GetWeights(round)                       → weights_cid, merkle_proof
Leave(contributor_id)                   → ack
```

### Shard lifecycle

```
ASSIGNED → SUBMITTED → MATCHED → VERIFIED → APPLIED
                    ↘
                      FLAGGED → REPROVED → VERIFIED
                             ↘
                               REJECTED
```

- `ASSIGNED`: given to a contributor with a timeout.
- `SUBMITTED`: `(delta_cid, ending_hash, sig)` received.
- `MATCHED`: `peer_k` submissions with identical `ending_hash`.
- `FLAGGED`: mismatched submissions; enters verifier queue.
- `VERIFIED`: matched or verifier-confirmed.
- `APPLIED`: delta consumed by the outer optimizer.
- `REJECTED`: verifier disagreement; submitter penalized; shard reassigned.

### Round

1. Coordinator freezes weights. Publishes CID + Merkle root.
2. Contributors pull weights and assigned data slices, run inner loops, submit deltas.
3. Coordinator accepts shards per lifecycle.
4. On `accepted_shards >= round_target` or `elapsed > round_timeout`, outer optimizer applies; round advances.

---

## 6. Trust

| Threat | Defense |
|---|---|
| Garbage delta | Peer hash disagreement. |
| Colluding contributors | Verifier re-execution at `spot_rate`. |
| Single coordinator replica compromise | Raft majority. |
| Data tampering | Content-addressed CIDs. |
| Plan tampering | Initiator signature. |
| Stalled contributor | Shard timeout; reputation penalty. |
| Malicious initiator | Contributor screens plan before joining. Registry tracks initiator reputation. |

Not defended:
- `>k/2` verifier collusion or `>⌊n/2⌋+1` coordinator replica compromise.
- Side channels on contributor hardware.
- Registry operator acting maliciously at the directory layer.

Verifier capacity must match `spot_rate · shards_total`. Undersized pools silently degrade the trust guarantee.

---

## 7. Failure Modes

| Failure | Handling |
|---|---|
| Contributor drops mid-shard | Timeout; reassignment; reputation decay. |
| Corrupted CID submission | Retrieval fails; reputation penalty. |
| Coordinator replica failure | Raft quorum continues; failed replica replays log. |
| Quorum loss | Run pauses. No advance until quorum restored. |
| Artifact store outage | Assignment blocks. |
| Verifier pool saturated | Spot rate temporarily reduced. Logged. |
| Initiator disappears | Run pauses at last applied round. In-flight shards may complete. |
| Nondeterministic contributor hardware | Routed to soft-verification if plan permits; else excluded. |

---

## 8. Reference Algorithms

### DiLoCo (default)

Inner: `H` steps of AdamW with seed `f(plan.seed_base, shard_id)` on the assigned data slice.

Outer:
```
Δ_i  = w_0 − w_i
Δ    = mean(Δ_i)
v_t  = μ · v_{t-1} + Δ       μ = 0.9
w_0 -= η · v_t               η = 0.7
```

### FedAvg

Outer step is the mean of contributor weights. Included for completeness.

### Custom

The runtime container implements:

```
outer_step(global_weights, [local_weights], round_state) → (global_weights', round_state')
```

Must be deterministic. Nondeterministic custom algorithms are not accepted.

---

## 9. Component Form Factors

| Component | Form | Rationale |
|---|---|---|
| Contributor agent | Native single-binary executable (Linux/macOS/Windows) + local web UI on `localhost:9000` | GPU training cannot run in a browser. Long-running daemon. Filesystem and container runtime access required. |
| Registry service | Web service + public web UI | Public directory. Browse and inspect without install. |
| Initiator tools (plan editor, run launcher) | Web app | Low frequency use. No GPU needed. Shareable plan templates. |
| Public dashboard (run explorer, ledger browser) | Web app | Public observability. Merkle proof verification runs client-side. |
| Coordinator | Server binary (headless) | Deployed by initiator as systemd service or container. |
| Verifier | Server binary (headless) | Deployed by verifier operators. |
| Reference runtime | Container image | Deterministic kernels pinned; portable across contributor OSes. |

The contributor agent auto-updates via signed releases and standard package managers. It runs as a background service across reboots.

---

## 10. Roadmap

### Phase 1 — MVP

Protocol implementation plus one demonstration run.

| Component | Stack |
|---|---|
| Registry | Rust, Axum, Postgres. Centralized, operated by Alchymia Labs. |
| Coordinator | Rust, openraft. Raft log = shard ledger. |
| Contributor agent | Rust CLI + local web UI. Orchestrates the reference runtime container. |
| Reference runtime | Container. Deterministic kernels. DiLoCo + AdamW only. |
| Verifier | Python. Re-runs the reference runtime. Fixed pool per plan. |
| Artifact store | S3, CID-addressed. |
| Dashboard | Web SPA. |

Work breakdown (estimates for a team of 2–3 engineers):

| Item | Duration |
|---|---|
| Deterministic reference runtime (H100, A100, Apple Silicon parity) | 4–6 weeks |
| Coordinator + Raft + shard ledger | 6–8 weeks |
| Registry service + UI | 3–4 weeks |
| Contributor agent (CLI + local UI + runtime orchestration) | 4–5 weeks |
| Verifier pool | 2 weeks |
| Public dashboard | 3–4 weeks |
| Integration, hardening, testing | 4–6 weeks |

Total: **4–6 months to acceptance** with 2–3 engineers. 10–14 months solo. Demonstration compute is separate.

Acceptance:
- Three concurrent runs on ≥10 contributors across ≥3 regions.
- Demonstration run converges within 3% of a centralized baseline on the same data.
- 30% mid-run contributor churn does not affect final loss.
- Shard inclusion is provable in `O(log N)` via Merkle path.
- A run replayed from the frozen ledger produces a bit-identical checkpoint.

Hardest parts:
- Bit-identical training across H100, A100, and Apple Silicon. Requires pinned kernels, deterministic reductions, fp16/bf16 mode fixed per run.
- Raft log as shard ledger — snapshot, rotation, replay edge cases.
- Verifier capacity balancing against submission rate under load.

### Phase 2 — Hardening

Production readiness and operator independence.

- IPFS replaces S3. Artifacts addressable without a single operator.
- Custom algorithm runtimes accepted. The `runtime_cid` field becomes normative.
- Horizontal coordinator scaling: shard ledger partitioning for runs with `shards_total` > 10⁶.
- Reputation portability: contributor reputation roams between registries.
- Observability: standardized metrics, incident runbook, SLOs for Registry and coordinator.

Acceptance:
- A run initiated by a non-Alchymia operator completes without Alchymia involvement beyond the Registry.
- A custom outer-loop algorithm executes end-to-end.
- Coordinator handles ≥10⁷ shards without degradation.

Timeline: **3–4 months** with 2–3 engineers.

### Phase 3 — Decentralized verification

Remove the verifier pool as a trust anchor.

- Zero-knowledge proofs of correct inner-loop execution. Submission includes a proof instead of a hash when `verification.mode = "zk"`.
- Open verifier participation with cryptoeconomic stake. Misbehaving verifiers lose stake.
- Plan-specified slashing conditions.

Acceptance:
- A run completes with `verification.mode = "zk"` and no trusted verifier pool.
- Slashing mechanism triggers correctly on simulated misconduct.

Timeline: **6–12 months**. Research-heavy. ZK proofs of ML training are an open problem; schedule depends on upstream progress (Polyhedra, zkML, Modulus).

### Phase 4 — Scale and heterogeneity

Support for frontier-scale runs and mixed contributor pools.

- Convergence at 100B+ parameters validated empirically.
- Adaptive `H` per contributor based on network conditions, bounded by determinism constraints.
- Cross-run scheduling: contributors with capacity split across concurrent runs report allocation intent; coordinators cooperate on reassignment.
- Mixed public/private data: plan-level data-access policies per contributor tier.

Acceptance:
- A ≥100B parameter run trained to completion on DSTP.
- Demonstrated adaptive `H` without determinism loss.
- Two concurrent high-demand runs share a contributor pool without starvation.

Timeline: **6–12 months**. Bottleneck is the empirical validation run, not engineering.

### Phase 5 — Federation

Registry decentralization.

- Multiple registry operators. Cross-registry run discovery.
- Plan and reputation portability across registries.
- Gossip protocol for run advertisements.
- Conflict resolution for contributor identity claimed by multiple registries.

Acceptance:
- Three independently operated registries participate in a shared contributor directory.
- A run registered on one registry is discoverable and joinable from another.

Timeline: **2–3 months** with 2–3 engineers.

### Total

Phases 1–2 deliver a production system with one operator. Phases 3–5 remove that operator and scale the protocol. Minimum viable path to a trust-minimized frontier-scale protocol: **~2 years**. Phase 1 alone is the MVP that enables real usage.

---

## 11. Unsolved

- Plan-level incentive enforcement without custody. Payment is off-protocol; DSTP offers the ledger but not the rails.
- Data privacy beyond public/private tiers. True secure multi-party training is out of scope.
- Byzantine fault tolerance above the Raft threshold.
- Cross-run scheduling fairness under adversarial contributor strategies.

---

## 12. References

- Douillard et al., 2023. DiLoCo. https://arxiv.org/abs/2311.08105
- Ryabinin et al., 2023. SWARM Parallelism. https://arxiv.org/abs/2301.11913
- Borzunov et al., 2022. Petals. https://arxiv.org/abs/2209.01188
- McMahan et al., 2017. Federated Averaging. https://arxiv.org/abs/1602.05629
- Prime Intellect, 2024. INTELLECT-1.
- Nous Research, 2024. DisTrO.
- Ongaro and Ousterhout, 2014. Raft.
