"""Crash-consistent per-epoch snapshots for the long Phase 1 fits.

Two mechanisms keep a long stage run from losing work between them:

* the batch runner stops *between* fits and keeps every packaged attempt
  durable, so completed fits survive anything;
* this module saves one snapshot at each completed epoch boundary, so a fit
  that dies mid-flight -- lease loss, OOM kill, wall-clock budget -- resumes
  from its last finished epoch instead of restarting from zero.

Resuming is only legitimate if it continues the *same* trajectory.  A resumed
fit that quietly diverged from the uninterrupted one would be a second,
unrecorded experiment.  The snapshot therefore carries everything the loop
reads on the way back in: model and optimizer state (including the learning
rate `adjust_learning_rate` has already set for the next epoch), the epoch
cursor, the early-stopping state, and the four RNG streams the loader's
shuffle order is drawn from.

What is deliberately *not* stored is any part of the frozen configuration.
The snapshot is keyed by the run's ``setting`` string, which already encodes
the model, data, split, shapes and seed, so a snapshot can never be replayed
under a different configuration -- and a training loop that finds a snapshot
its command line does not match fails loudly rather than resuming into it.

Networks are not involved here, and the file is written by our own training
process, so the load is a plain pickle.
"""

import os
import random

import numpy as np
import torch

SNAPSHOT_VERSION = 1
SNAPSHOT_NAME = "recovery.pt"


def snapshot_path(checkpoint_dir):
    """The snapshot lives beside ``checkpoint.pth``, under the run's setting."""
    return os.path.join(checkpoint_dir, SNAPSHOT_NAME)


def capture_rng():
    """Every stream the training loop advances, in one payload.

    The loader shuffles from the global torch stream; the protocol seeds
    python, numpy and torch identically, and CUDA keeps its own generator.
    All four travel together so that the epoch after a resume draws the same
    batch order as the epoch after an uninterrupted run.
    """
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "the snapshot holds CUDA RNG state but this process has no CUDA; "
                "resuming here would draw a different batch order"
            )
        torch.cuda.set_rng_state_all(state["cuda"])


def write_snapshot(path, payload):
    """Publish a snapshot atomically, so a kill can never leave a torn file.

    A replacement that is interrupted mid-write must leave the previous
    snapshot intact -- that copy is the only record of the epochs already
    paid for.
    """
    temporary = "{}.{}.tmp".format(path, os.getpid())
    with open(temporary, "wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_snapshot(path, identity):
    """Return the snapshot for ``identity``, or ``None`` when there is none."""
    if not os.path.isfile(path):
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    version = payload.get("snapshot_version")
    if version != SNAPSHOT_VERSION:
        raise RuntimeError(
            "snapshot {} has version {!r}; this build writes version {}".format(
                path, version, SNAPSHOT_VERSION
            )
        )
    if payload.get("identity") != identity:
        raise RuntimeError(
            "snapshot {} belongs to a different run than this command line: "
            "{!r} != {!r}".format(path, payload.get("identity"), identity)
        )
    return payload


def discard_snapshot(path):
    """Drop the snapshot once training has finished.

    The fit's result is in ``checkpoint.pth`` from here on, so a surviving
    snapshot would only risk resuming a run that already completed.
    """
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
