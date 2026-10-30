# Running on Slurm

`blut recipe run --launcher slurm` moves stage placement onto a Slurm cluster.
The recipe does not change; the allocation comes from environment variables, so
the same command runs locally or on the cluster.

## Requirements

- **A shared filesystem.** Job directories and the content-addressed cache must
  be on a filesystem every node mounts at the same path (NFS, Lustre, GPFS). A
  stage's inputs are passed to the compute nodes by path, not copied.
- **Slurm client commands on the submit host:** `srun`, `sbatch`, `scontrol`,
  `scancel`. `recipe run --launcher slurm` refuses to start when `sbatch` or
  `scontrol` is not on `PATH`. Accounting (`slurmdbd`, `sacct`) is optional. Without it, a job's
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

## Resuming after a lost node

A training stage that opts in to checkpoint slots (CONTRACT.md §8; the standard
cookbook's training stages do) resumes from its newest checkpoint instead of
step 0. Three ways to get there:

- **Requeue.** Submit the whole recipe as a requeueable batch job, so the
  training step is an `srun` inside its allocation:

  ```sh
  #!/bin/bash
  #SBATCH --nodes=2 --ntasks-per-node=1 --requeue --open-mode=append
  blut-core recipe run train_from_dataset --launcher slurm --args "$ARGS"
  ```

  When a node fails, Slurm requeues the job and the same command runs again as
  a new BLUT job. The training stage has the same cache key, so it finds the
  same slot. The engine reads `SLURM_JOB_ID` and `SLURM_RESTART_COUNT` and
  treats a slot left by an earlier run of the same job as abandoned, so the
  requeued run resumes without waiting for the old heartbeat to go stale.
- **`plan resume`.** After the BLUT process or its host dies, `blut-core plan
  resume <job>` reruns the plan; finished stages come from the cache and the
  training stage resumes. Killing only the BLUT process does not stop the
  Slurm job its `srun` started: the training keeps running. The job directory
  records the name of every Slurm job its stages started
  (`blut-<job id>-<node index>`, from `StageContext::slurm_job_name`), so
  `plan resume` refuses while one is still queued or running and names it,
  and `blut cancel <job>` `scancel`s it, also after the BLUT process is gone.
  Cancel, then resume.
- **Retry.** `train_attempts: 2` (or more) retries the training stage in the
  same BLUT process. A retry's `srun` waits until enough nodes are back, then
  resumes.

Checkpoints live under `$BLUT_CHECKPOINTS_DIR`, by default beside the jobs
directory. Only rank 0 reads and writes them, so it needs to be on rank 0's
node; on a cluster, put it on the shared filesystem with the job directories.

When an attempt fails under `--launcher slurm`, the engine cancels the Slurm
job of that attempt by name: killing `srun` does not end the job, and a retry
must not share the nodes with a leftover trainer. A stage that names its job
some other way should record the name with `jobs::record_slurm_job_name`.

## What has been run

[`DISTRIBUTED_VALIDATION.md`](DISTRIBUTED_VALIDATION.md) records the runs:
- a two-node Slurm 21.08 cluster, with and without accounting;
- two-machine NCCL training through the cookbook;
- a training stage resuming from its checkpoint after a requeue, a `plan
  resume`, and a retry, on two machines.

`tests/slurm_live.rs` repeats the cluster checks on any cluster:

```sh
BLUT_SLURM_LIVE=1 cargo test --test slurm_live -- --ignored --test-threads=1
```

Run it from a directory on the shared filesystem, with two idle nodes.
