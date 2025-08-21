#!/usr/bin/env python3
############ Imports ###################
import os
import uuid
import json
import time
import random
import asyncio
import logging
import math
from typing import Dict, Optional, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from azure.storage.blob import BlobServiceClient

############ Logging ###################
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("bfd-api")

app = FastAPI()

# In-memory job tracking
job_states: Dict[str, Dict] = {}
job_results: Dict[str, Dict] = {}

############# Request model #############
class RunRequest(BaseModel):
    # Problem size and platform parameters
    num_tasks: int        = Field(default=int(os.getenv("NUM_TASKS", 1000)), gt=0)
    num_cores: int        = Field(default=int(os.getenv("NUM_CORES", 16)), gt=1)

    # Energy model
    base_energy: float    = Field(default=float(os.getenv("BASE_ENERGY", 0.01)), ge=0)
    idle_energy: float    = Field(default=float(os.getenv("IDLE_ENERGY", 0.002)), ge=0)

    # Synthetic workload generation
    seed: int             = Field(default=int(os.getenv("SEED", 42)))
    min_exec: int         = Field(default=int(os.getenv("MIN_EXEC", 10)), ge=0)
    max_exec: int         = Field(default=int(os.getenv("MAX_EXEC", 20)), ge=0)


############ Core BFD logic ############
def _bfd_softcap_multiobjective(exec_times: List[float], num_cores: int, base_energy: float, idle_energy: float):
    """
    Best-Fit Decreasing with a penalty-based soft cap (no hard capacity).
    score = 0.4*(makespan/S) + 0.2*(total_e/S) + 0.4*imbalance
      where S = sum(exec_times) or 1
            total_e = sum(ct*base_energy) + sum((makespan - ct)*idle_energy)
            imbalance = pstdev(core_times) / mean(core_times) (population)
    Returns (task_to_core, core_times, final_metrics_dict, progress_scores)
    """
    n = len(exec_times)
    m = num_cores
    S = sum(exec_times) or 1.0

    # Sort task indices by descending duration (stable by index)
    order = sorted(range(n), key=lambda i: (-exec_times[i], i))

    core_times = [0.0] * m
    task_to_core = [-1] * n

    # Aggregates for O(1) trial scoring
    sum_ct = 0.0
    sumsq_ct = 0.0
    makespan = 0.0

    # Track score after each placement
    progress_scores: List[float] = []

    def _score_trial(x_old: float, add_t: float, M_curr: float, sum_ct_curr: float, sumsq_ct_curr: float) -> float:
        new_x = x_old + add_t
        sum_prime = sum_ct_curr + add_t
        sumsq_prime = sumsq_ct_curr + 2.0 * x_old * add_t + add_t * add_t
        M_prime = new_x if new_x > M_curr else M_curr

        # Energy components
        active_e = base_energy * sum_prime
        idle_e   = idle_energy * (m * M_prime - sum_prime)
        total_e  = active_e + idle_e

        # Imbalance (population stddev / mean)
        mean_ct = sum_prime / m if m > 0 else 0.0
        if mean_ct == 0.0:
            imbalance = 0.0
        else:
            var = (sumsq_prime / m) - (mean_ct * mean_ct)
            if var < 0.0:
                var = 0.0
            pstdev_v = math.sqrt(var)
            imbalance = pstdev_v / mean_ct

        score = (0.4 * (M_prime / S)) + (0.2 * (total_e / S)) + (0.4 * imbalance)
        return score

    def _score_current(sum_ct_curr: float, sumsq_ct_curr: float, M_curr: float) -> float:
        active_e = base_energy * sum_ct_curr
        idle_e   = idle_energy * (m * M_curr - sum_ct_curr)
        total_e  = active_e + idle_e
        mean_ct = (sum_ct_curr / m) if m > 0 else 0.0
        if mean_ct == 0.0:
            imbalance = 0.0
        else:
            var = (sumsq_ct_curr / m) - (mean_ct * mean_ct)
            if var < 0.0:
                var = 0.0
            pstdev_v = math.sqrt(var)
            imbalance = pstdev_v / mean_ct
        score = (0.4 * (M_curr / S)) + (0.2 * (total_e / S)) + (0.4 * imbalance)
        return score

    # Greedy placement: for each task, pick the core that minimizes the score
    for t_idx in order:
        t = exec_times[t_idx]
        best_core = 0
        best_score = float("inf")

        for c in range(m):
            x = core_times[c]
            score_prime = _score_trial(x, t, makespan, sum_ct, sumsq_ct)
            if score_prime < best_score or (score_prime == best_score and c < best_core):
                best_score = score_prime
                best_core = c

        # Commit placement
        old_x = core_times[best_core]
        new_x = old_x + t
        core_times[best_core] = new_x
        sum_ct += t
        sumsq_ct += 2.0 * old_x * t + t * t
        if new_x > makespan:
            makespan = new_x
        task_to_core[t_idx] = best_core

        # Record current score after this placement
        progress_scores.append(_score_current(sum_ct, sumsq_ct, makespan))

    # Final metrics from aggregates
    active_e = base_energy * sum_ct
    idle_e   = idle_energy * (m * makespan - sum_ct)
    total_e  = active_e + idle_e
    mean_ct = (sum_ct / m) if m > 0 else 0.0
    if mean_ct == 0.0:
        imbalance = 0.0
    else:
        var = (sumsq_ct / m) - (mean_ct * mean_ct)
        if var < 0.0:
            var = 0.0
        pstdev_v = math.sqrt(var)
        imbalance = pstdev_v / mean_ct

    score = (0.4 * (makespan / S)) + (0.2 * (total_e / S)) + (0.4 * imbalance)

    final = {
        "score": score,
        "makespan": makespan,
        "total_e": total_e,
        "imbalance": imbalance,
        "core_times": core_times,
        "best_individual": task_to_core
    }
    return task_to_core, core_times, final, progress_scores

########## API Endpoints ##############

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.post("/run")
async def run(req: RunRequest):
    job_id = str(uuid.uuid4())
    job_states[job_id] = {"status": "running"}
    asyncio.create_task(_run_bfd(job_id, req))
    return {"job_id": job_id}

@app.get("/result/{job_id}")
async def result(job_id: str):
    state = job_states.get(job_id)
    if state is None:
        raise HTTPException(404, "Job not found")
    if state.get("status") == "running":
        raise HTTPException(400, "Job still running")
    res = job_results.get(job_id)
    if not res:
        raise HTTPException(404, "Job not found or failed")
    return res

############ Internal BFD runner ############
async def _run_bfd(job_id: str, req: RunRequest):
    # Storage config is supplied via environment (manifests)
    conn_str  = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    container = os.getenv("BLOB_CONTAINER")
    if not conn_str or not container:
        job_states[job_id]["status"] = "error"
        job_states[job_id]["error"]  = "Missing AZURE_STORAGE_CONNECTION_STRING or BLOB_CONTAINER"
        return

    logger.info(
        f"[{job_id}] BFD params: tasks={req.num_tasks}, cores={req.num_cores}, "
        f"baseE={req.base_energy}, idleE={req.idle_energy}, "
        f"seed={req.seed}, range=[{req.min_exec},{req.max_exec}]"
    )

    # Deterministic synthetic workload
    rng_exec = random.Random(req.seed)
    lo, hi = int(req.min_exec), int(req.max_exec)
    if hi < lo:
        lo, hi = hi, lo
    exec_times = [float(rng_exec.randint(lo, hi)) for _ in range(req.num_tasks)]

    start_time = time.time()

    # Run BFD and update status as tasks are placed
    task_to_core, core_times, final_metrics, progress = _bfd_softcap_multiobjective(
        exec_times, req.num_cores, req.base_energy, req.idle_energy
    )

    elapsed = time.time() - start_time

    # Final result bundle
    final = {
        # Identity
        "run_id":               job_id,
        # Config snapshot
        "num_tasks":            req.num_tasks,
        "num_cores":            req.num_cores,
        "base_energy":          req.base_energy,
        "idle_energy":          req.idle_energy,
        "seed":                 req.seed,
        # Outcomes
        "elapsed_time_s":       elapsed,
        "best_individual":      final_metrics["best_individual"],
        "best_fitness":         final_metrics["score"],
        # Raw metrics for the final solution
        "makespan":             final_metrics["makespan"],
        "total_energy":         final_metrics["total_e"],
        "imbalance":            final_metrics["imbalance"],
        "core_times":           final_metrics["core_times"],
        "exec_times":           exec_times,
        "created_at":           time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }

    # Upload JSON
    svc = BlobServiceClient.from_connection_string(conn_str)
    try:
        svc.create_container(container)
    except Exception:
        pass
    txt_blob = svc.get_blob_client(container=container, blob=f"{job_id}.txt")
    txt_blob.upload_blob(json.dumps(final), overwrite=True)

    job_results[job_id] = final
    job_states[job_id] = {"status": "done"}