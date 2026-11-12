"""Compare a sequential cascade with trial-parallel Spark scheduling.

The three input files must contain the same zero-based channel_ids and fsamp.
Times are seconds from the original aligned trial start.
"""

import argparse
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np

LEVELS = [(3, 9), (13, 19), (23, 29)]
FS = 2048
EXT = 10
SIL = 0.8
PNR = 15.0


def read_emg(path):
    with np.load(path, allow_pickle=False) as data:
        if int(data["fsamp"]) != FS:
            raise ValueError(f"Sampling rate must be {FS}: {path}")
        return data["emg"].astype(np.float32), data["channel_ids"]


def task(spec):
    import torch

    from mu_course.ckc.cascade_decomposer import SegmentDecomposer, refine_filters

    torch.set_num_threads(spec["threads"])
    np.random.seed(0)
    torch.manual_seed(0)
    started = time.time()
    emg, _ = read_emg(spec["path"])
    start, end = LEVELS[spec["level"]]
    emg = emg[:, start * FS : end * FS]
    if emg.shape[1] != (end - start) * FS:
        raise ValueError("Trial does not contain the requested plateau")
    if spec["track"] == "A":
        model = SegmentDecomposer(
            seg_duration_s=15,
            fsamp=FS,
            ext_factor=EXT,
            n_mu_max=spec["n_mu_max"],
            n_iter=45,
            cost_fcn=1,
            sil_threshold=SIL,
            pnr_threshold_db=PNR,
            device="cpu",
        )
        model.fit(emg, emg_full=emg)
        vectors = np.asarray(model.mu_templates, dtype=np.float32).reshape(-1, emg.shape[0] * EXT)
        pnrs = np.asarray(model.pnrs, dtype=float)
        lineages = (
            np.arange(len(vectors), dtype=np.int64) + spec["level"] * 100000 + spec["trial"] * 10000
        )
    else:
        pool = spec["pool"]
        if len(pool["vectors"]):
            result = refine_filters(
                pool["vectors"],
                emg,
                fsamp=FS,
                ext_factor=EXT,
                n_iter=5,
                cost_fcn=1,
                sil_threshold=SIL,
                device="cpu",
                verbose=False,
            )
            keep = np.asarray(
                [
                    s > SIL and len(p) > 3 and n > PNR
                    for s, p, n in zip(result["sils"], result["mu_pulses"], result["pnrs"])
                ]
            )
            vectors = result["refined_filters"][keep]
            pnrs = np.asarray(result["pnrs"])[keep]
            lineages = pool["lineages"][keep]
        else:
            vectors = np.empty((0, emg.shape[0] * EXT), dtype=np.float32)
            pnrs = np.empty(0)
            lineages = np.empty(0, dtype=np.int64)
    if len(vectors) != len(pnrs):
        raise ValueError("Template and PNR counts do not match")
    keep = np.isfinite(pnrs) & (pnrs > PNR)
    return {
        "vectors": vectors[keep],
        "pnrs": pnrs[keep],
        "lineages": lineages[keep],
        "track": spec["track"],
        "trial": spec["trial"],
        "level": spec["level"],
        "started": started,
        "finished": time.time(),
        "pid": os.getpid(),
    }


def cosine_indices(vectors, pnrs, threshold=0.95):
    if not len(vectors):
        return np.empty(0, dtype=int)
    normalized = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10)
    similarity = np.abs(normalized @ normalized.T)
    keep = []
    for i in np.argsort(-pnrs, kind="stable"):
        if all(similarity[i, j] < threshold for j in keep):
            keep.append(i)
    return np.asarray(keep, dtype=int)


def merge(results, paths):
    import torch

    from mu_course.ckc.cascade_decomposer import _rate_of_agreement, refine_filters

    torch.set_num_threads(1)
    started = time.time()
    # Research pipeline first deduplicates A+B within each trial.
    trial_pools = []
    for trial in range(len(paths)):
        group = [r for r in results if r["trial"] == trial]
        vectors = np.concatenate([r["vectors"] for r in group])
        pnrs = np.concatenate([r["pnrs"] for r in group])
        lineages = np.concatenate([r["lineages"] for r in group])
        idx = cosine_indices(vectors, pnrs)
        trial_pools.append((vectors[idx], pnrs[idx], lineages[idx]))
    vectors = np.concatenate([p[0] for p in trial_pools])
    pnrs = np.concatenate([p[1] for p in trial_pools])
    lineages = np.concatenate([p[2] for p in trial_pools])
    seen, idx = set(), []
    for i in np.argsort(-pnrs, kind="stable"):
        if int(lineages[i]) not in seen:
            seen.add(int(lineages[i]))
            idx.append(i)
    vectors, pnrs, lineages = vectors[idx], pnrs[idx], lineages[idx]
    idx = cosine_indices(vectors, pnrs)
    vectors, pnrs, lineages = vectors[idx], pnrs[idx], lineages[idx]

    # Match the research implementation: it skips both source filters for <=1 MU.
    if len(vectors) > 1:
        pulses_by_trial = []
        for path in paths:
            emg, _ = read_emg(path)
            result = refine_filters(
                vectors,
                emg,
                fsamp=FS,
                ext_factor=EXT,
                n_iter=0,
                cost_fcn=1,
                sil_threshold=SIL,
                device="cpu",
                verbose=False,
            )
            pulses_by_trial.append(result["mu_pulses"])
        norm = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10)
        sim = np.abs(norm @ norm.T)
        kept = []
        for i in np.argsort(-pnrs, kind="stable"):
            duplicate = False
            for j in kept:
                if sim[i, j] < 0.90:
                    continue
                agree = 0
                for pulses in pulses_by_trial:
                    a, b = pulses[i], pulses[j]
                    ratio = len(a) / max(len(b), 1)
                    if (
                        0.3 <= ratio <= 3.3
                        and _rate_of_agreement(a, b, int(FS * 0.1), int(FS / 100)) > 0.5
                    ):
                        agree += 1
                if agree >= 2:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(i)
        rest_keep = []
        for i in kept:
            flagged = 0
            for pulses in pulses_by_trial:
                count = sum(
                    np.sum((pulses[i] >= int(a * FS)) & (pulses[i] < int(b * FS)))
                    for a, b in [(0.5, 1.5), (10.5, 11.5), (20.5, 21.5)]
                )
                flagged += count > 15
            if flagged < 2:
                rest_keep.append(i)
        vectors, pnrs, lineages = vectors[rest_keep], pnrs[rest_keep], lineages[rest_keep]
    return {
        "vectors": vectors,
        "pnrs": pnrs,
        "lineages": lineages,
        "started": started,
        "finished": time.time(),
        "pid": os.getpid(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["cpu", "spark"], required=True)
    parser.add_argument("--trials", nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-mu-max", type=int, default=40)
    parser.add_argument("--slots", type=int, default=3, help="Spark local task slots")
    parser.add_argument("--partitions", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1, help="Torch threads per task")
    args = parser.parse_args()
    if min(args.n_mu_max, args.slots, args.partitions, args.threads) < 1:
        parser.error("Candidate count, slots, partitions and threads must be positive")
    paths = [str(Path(p).resolve()) for p in args.trials]
    channels = [read_emg(p)[1] for p in paths]
    if not all(np.array_equal(channels[0], c) for c in channels[1:]):
        raise ValueError("Input files must have identical channel IDs in the same order")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    events = []
    params = {
        "mode": args.mode,
        "trials": paths,
        "channel_ids": channels[0].tolist(),
        "levels_s": LEVELS,
        "n_mu_max": args.n_mu_max,
        "n_iter": 45,
        "refine_iter": 5,
        "fsamp": FS,
        "ext_factor": EXT,
        "sil_threshold": SIL,
        "pnr_threshold_db": PNR,
        "device": "cpu",
        "slots": args.slots if args.mode == "spark" else None,
        "partitions": args.partitions if args.mode == "spark" else None,
        "task_threads": args.threads,
        "merge_threads": 1,
        "cpu_affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
    }
    (args.output / "parameters.json").write_text(json.dumps(params, indent=2), encoding="utf-8")

    def specs(level, track, pool=None):
        return [
            {
                "level": level,
                "track": track,
                "trial": i,
                "path": p,
                "pool": pool,
                "n_mu_max": args.n_mu_max,
                "threads": args.threads,
            }
            for i, p in enumerate(paths)
        ]

    def record(items):
        for item in items:
            events.append(
                {k: item[k] for k in ("track", "trial", "level", "started", "finished", "pid")}
            )
        (args.output / "events.json").write_text(json.dumps(events, indent=2), encoding="utf-8")

    def spark_tasks(sc, items):
        return sc.parallelize(items, args.partitions).map(task).collect()

    if args.mode == "cpu":
        pool = None
        for level in range(len(LEVELS)):
            results = [task(s) for s in specs(level, "A")]
            if pool is not None:
                results += [task(s) for s in specs(level, "B", pool)]
            record(results)
            pool = merge(results, paths)
            events.append(
                {
                    "track": "merge",
                    "level": level,
                    "started": pool["started"],
                    "finished": pool["finished"],
                    "pid": pool["pid"],
                }
            )
            np.savez(args.output / f"pool_{level}.npz", **pool)
            (args.output / "events.json").write_text(json.dumps(events, indent=2), encoding="utf-8")
    else:
        # Local task slots and a separate merge process share this host.
        os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
        from pyspark.sql import SparkSession

        with (
            ProcessPoolExecutor(
                max_workers=1, mp_context=multiprocessing.get_context("spawn")
            ) as merger,
            ThreadPoolExecutor(max_workers=1) as runner,
        ):
            spark = (
                SparkSession.builder.master(f"local[{args.slots}]")
                .appName("cascade-comparison")
                .getOrCreate()
            )
            sc = spark.sparkContext
            try:
                current = spark_tasks(sc, specs(0, "A"))
                record(current)
                for level in range(len(LEVELS)):
                    merge_future = merger.submit(merge, current, paths)
                    next_a = (
                        runner.submit(spark_tasks, sc, specs(level + 1, "A"))
                        if level + 1 < len(LEVELS)
                        else None
                    )
                    pool = merge_future.result()
                    events.append(
                        {
                            "track": "merge",
                            "level": level,
                            "started": pool["started"],
                            "finished": pool["finished"],
                            "pid": pool["pid"],
                        }
                    )
                    np.savez(args.output / f"pool_{level}.npz", **pool)
                    (args.output / "events.json").write_text(
                        json.dumps(events, indent=2), encoding="utf-8"
                    )
                    if next_a is not None:
                        a_results = next_a.result()
                        b_results = spark_tasks(sc, specs(level + 1, "B", pool))
                        record(a_results + b_results)
                        current = a_results + b_results
            finally:
                spark.stop()
    (args.output / "events.json").write_text(json.dumps(events, indent=2), encoding="utf-8")
    (args.output / "summary.json").write_text(
        json.dumps(
            {"elapsed_seconds": time.time() - started, "final_mu_count": len(pool["vectors"])},
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
