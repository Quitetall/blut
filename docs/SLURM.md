# Running on Slurm

`blut recipe run --launcher slurm` moves stage placement onto a Slurm cluster.
The recipe does not change; the allocation comes from environment variables, so
the same command runs locally or on the cluster.

## Requirements

- **A shared filesystem.** Job directories and the content-addressed cache must
  be on a filesystem every node mounts at the same path (NFS, Lustre, GPFS). A
  stage's inputs are passed to the compute nodes by path, not copied.
- **Slurm client commands on the submit host:** `srun`, `sbatch`, `scontrol`,
  `scancel`. Accounting (`slurmdbd`, `sacct`) is optional. Without it, a job's
  state can be read until the controller purges the record (`MinJobAge`,
  300 s by default). After that, polling reports that the job is unknown.

## Allocation

| Variable | Slurm flag | Meaning |
|---|---|---|
| `BLUT_SLURM_PARTITION` | `--partition` | Partition to submit to. |
| `BLUT_SLURM_TIME` | `--time` | Wall-clock limit, e.g. `08:00:00`. This is what bounds a job: BLUT waits as long as the job runs. |
| `BLUT_SLURM_MEM` | `--mem` | Memory per node, e.g. `44G`. |
| `BLUT_SLURM_CPUS` | `--cpus-per-task` | CPUs per task. |
| `BLUT_SLURM_GPUS` | `--gpus` | **Total** GPUs for the job, across all of its nodes. BLUT also uses it as the number of GPUs it may schedule at once. |
| `BLUT_SLURM_NODES` | `--nodes` | Number of nodes. |
| `BLUT_SLURM_NTASKS_PER_NODE` | `--ntasks-per-node` | Tasks per node. Slurm's default is one. |

A numeric variable that is not a positive integer stops the run before anything
starts, instead of being ignored.

`--gpus` is a total, not a per-node count. For four GPUs on each of two nodes,
set `BLUT_SLURM_GPUS=8`, or let a cookbook request `--gpus-per-node=4`. Slurm
accepts both when they agree (`8` = 2 nodes × 4) and refuses the job at
submission when they don't.

## Multi-node jobs

When a job has more than one task, the command runs as a job step on every task,
with this environment:

- `MASTER_ADDR`: the allocation's first host, unless you set it yourself.
- `MASTER_PORT`: `29500`, unless you set it yourself.
- `NODE_RANK`: the task's own node index (`SLURM_NODEID`), read inside the task.

That is the environment `torchrun --rdzv-backend=c10d` needs. One task per node,
with `torchrun` starting the per-GPU processes, is the usual shape.

Training stages live in cookbooks. The standard cookbook's `train_from_dataset`
recipe has been run this way across two machines; its README covers the
training-side options.

## Logs

`sbatch` jobs write to `slurm-<jobid>.out` in the directory BLUT was started
from. BLUT streams that file while the job runs. An `--output` or `--chdir`
passed through `extra` is honoured. If the `--output` pattern names one file per
node or task (`%N`, `%t`), BLUT says so and does not stream the log.

## What has been run

[`DISTRIBUTED_VALIDATION.md`](DISTRIBUTED_VALIDATION.md) records the runs:
- a two-node Slurm 21.08 cluster, with and without accounting;
- two-machine NCCL training through the cookbook.

`tests/slurm_live.rs` repeats the cluster checks on any cluster:

```sh
BLUT_SLURM_LIVE=1 cargo test --test slurm_live -- --ignored --test-threads=1
```

Run it from a directory on the shared filesystem, with two idle nodes.
