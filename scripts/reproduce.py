#!/usr/bin/env python3
"""Re-run the preprint's simulations or rebuild the paper from archived results."""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/hbf_adoption"
PAPER = ROOT / "paper"
RUNS = ROOT / "runs"
BUILD = ROOT / ".cache/hbf-adoption/compiled-event-model"
PROOF = BUILD.parent / "compiled-check"


def python(script, *args):
    return [sys.executable, script, *args]


def launch(argv, **kwargs):
    subprocess.run([str(arg) for arg in argv], check=True, **kwargs)


def simulate(script, *args):
    return python(EXPERIMENT / script, *args)


def bounded(script, *args):
    return simulate("memory_entry.py", EXPERIMENT / script, *args)


def save(path, data):
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(data, indent=2) + "\n")
    pending.replace(path)


@contextmanager
def lock(folder):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "run.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"A suite is already running in {folder}") from None
        yield


class Suite:
    def __init__(self, folder):
        self.folder = folder
        self.status = folder / "status.json"
        inputs = [Path(__file__), ROOT / "requirements.txt", PAPER / "make_figures.py",
                  *EXPERIMENT.glob("*.py"), *EXPERIMENT.glob("*.md"),
                  *(p for p in (EXPERIMENT / "results").rglob("*") if p.is_file())]
        digest = hashlib.sha256()
        for path in sorted(inputs):
            digest.update(str(path.relative_to(ROOT)).encode())
            digest.update(hashlib.sha256(path.read_bytes()).digest())
        digest.update(sys.version.encode())
        self.state = {"inputs_sha256": digest.hexdigest(), "stages": {}}
        if self.status.exists():
            previous = json.loads(self.status.read_text())
            if previous["inputs_sha256"] != self.state["inputs_sha256"]:
                raise SystemExit("Inputs changed. Select a new --name to start a separate run.")
            self.state = previous

    def step(self, name, command, *, incremental=False, output_file=None):
        steps = self.state["stages"]
        previous = steps.get(name, {})
        parent = self.folder / name
        if previous.get("status") == "complete":
            destination = self.folder / previous["output"]
            if destination.exists():
                print(f"{name} already complete", flush=True)
                return destination
            raise SystemExit(f"{destination} is missing. Select a new --name to rerun the suite.")
        parent.mkdir(exist_ok=True)
        logs = sorted(parent.glob("attempt-*.log"))
        log = parent / f"attempt-{len(logs)+1:04}.log"
        output = (parent / "data" if incremental else log.with_suffix(""))
        if output_file:
            output = output.with_suffix(output_file)
        relative = lambda path: str(path.relative_to(self.folder))
        steps[name] = dict(status="running", output=relative(output), log=relative(log))
        save(self.status, self.state)
        print(f"{name} running. Log in {log.relative_to(ROOT)}", flush=True)
        with log.open("x") as stream:
            try:
                launch(command(output), stdout=stream, stderr=subprocess.STDOUT)
            except KeyboardInterrupt:
                steps[name]["status"] = "interrupted"
                save(self.status, self.state)
                raise
            except subprocess.CalledProcessError:
                steps[name]["status"] = "failed"
                save(self.status, self.state)
                raise
        steps[name]["status"] = "complete"
        save(self.status, self.state)
        return output

    def full(self, workers):
        def central_command(out):
            previous = [folder for folder in sorted(out.parent.glob("attempt-*"))
                        if (folder / "manifest.json").is_file()]
            return simulate("parallel_projection_runs.py", "--out", out, "--workers", workers,
                            "--matched-domain", "--low-memory", "--event-budget", 2_000_000_000,
                            *(item for folder in previous for item in ("--completed-from", folder)))

        central = self.step("central", central_command)
        expansion = self.step("fixed-demand", lambda out: simulate(
            "fixed_demand_parallel.py", "--out", out, "--workers", workers, "--low-memory"))
        corrections = self.step("boundary", lambda out: bounded(
            "boundary_replay.py", "--out", out, "--input", central))
        corners = self.step("sensitivity", lambda out: simulate(
            "sensitivity_runs.py", "--out", out, "--input", central, "--expansion", expansion,
            "--workers", workers, "--low-memory"))
        routes = self.step("routing", lambda out: bounded("route_intervention.py", "--out", out))
        muse_matrix = self.step("muse", lambda out: simulate(
            "muse_spark_matrix.py", "--out", out, "--workers", workers), incremental=True)
        common = ("--corrections", corrections, "--expansion", expansion)
        adoption = self.step("adoption", lambda out: simulate(
            "analyze_deployments.py", "--input", central, "--out", out, *common))
        uncertainty = self.step("uncertainty", lambda out: simulate(
            "analyze_sensitivity.py", "--input", corners, "--routes", routes, "--central", central,
            "--report", adoption, "--out", out, *common))
        muse = self.step("muse-analysis", lambda out: simulate(
            "analyze_muse.py", "--matrix", muse_matrix, "--out", out), output_file=".json")
        results = self.folder / "results"
        results.mkdir(exist_ok=True)
        for name, source in (("adoption", adoption), ("uncertainty", uncertainty), ("muse.json", muse)):
            link = results / name
            if not link.exists():
                link.symlink_to(os.path.relpath(source, results))
        self.step("muse-timing-audit", lambda out: simulate(
            "muse_spark_timing_audit.py", "--out", out), output_file=".json")
        self.step("muse-crosscheck", lambda out: simulate(
            "muse_projection_crosscheck.py", "--out", out), output_file=".json")
        figures = self.step("figures", lambda out: python(
            PAPER / "make_figures.py", "--results", results, "--out", out), incremental=True)
        print(f"Results in {results.relative_to(ROOT)}. Figures in {figures.relative_to(ROOT)}.", flush=True)

    def smoke(self):
        self.step("central", lambda out: bounded(
            "matched_projection_sweep.py", "--out", out, "--models", "qwen3_8b", "--gpus", "h200",
            "--counts", 1, "--workloads", "short_chat", "--views", "fixed", "--seeds", 17))
        self.step("muse", lambda out: simulate(
            "muse_spark_sweep.py", "--out", out, "--gpu", "h200", "--counts", 1,
            "--workload", "short_chat", "--views", "fixed", "--seeds", 17))


def show_status(folder):
    status = folder / "status.json"
    if not status.exists():
        print(f"No recorded run in {folder}")
        return
    state = json.loads(status.read_text())
    for name, step in state["stages"].items():
        path = folder / step["output"] / "manifest.json"
        try:
            manifest = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if name == "central" and "jobs" in manifest:
            step["worker_jobs_completed"] = sum(job["status"] == "pass" for job in manifest["jobs"])
            step["worker_jobs_expected"] = 16
        elif name == "fixed-demand":
            step["cells_completed"] = manifest["cells"]
            step["cells_expected"] = 720
        elif name == "sensitivity":
            step["cases_completed"] = sum(job["status"] == "pass" for job in manifest["jobs"])
            step["cases_expected"] = 6
        elif "cells_completed" in manifest and "cells_expected" in manifest:
            step["cells_completed"] = manifest["cells_completed"]
            step["cells_expected"] = manifest["cells_expected"]
    print(json.dumps(state, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("smoke", "full", "status", "tests", "compile", "figures", "paper"))
    parser.add_argument("--name", help="Output directory name under runs/ for full, smoke or status")
    parser.add_argument("--workers", type=int, default=2, help="Parallel simulator processes for full")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    name = args.name or ("smoke" if args.action == "smoke" else "full")
    if name in (".", "..") or Path(name).name != name:
        parser.error("--name must be a single directory name")
    folder = RUNS / name
    os.chdir(ROOT)
    mpl = ROOT / ".cache/matplotlib"
    mpl.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl))
    os.environ.setdefault("MPLBACKEND", "Agg")

    try:
        if args.action in ("smoke", "full"):
            with lock(folder):
                suite = Suite(folder)
                if args.action == "full":
                    suite.full(args.workers)
                else:
                    suite.smoke()
        elif args.action == "status":
            show_status(folder)
        elif args.action == "tests":
            launch(python("-m", "unittest", "discover", "-s", EXPERIMENT, "-v"))
        elif args.action == "compile":
            if not BUILD.exists():
                launch(simulate("build_compiled_runtime.py", "--out", BUILD))
            if not (PROOF / "manifest.json").exists():
                if PROOF.exists():
                    shutil.rmtree(PROOF)
                launch(simulate("check_compiled_runtime.py", "--out", PROOF, "--tests"))
            launch(python("-c", "from compiled_entry import check; check()"), cwd=EXPERIMENT)
        elif args.action == "figures":
            launch(python(PAPER / "make_figures.py", "--out", RUNS / "figures"))
        elif args.action == "paper":
            launch(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
                    f"-outdir={RUNS / 'paper'}", "main.tex"], cwd=PAPER)
    except KeyboardInterrupt:
        return 130
    except subprocess.CalledProcessError as error:
        print(f"Command failed with exit code {error.returncode}.", file=sys.stderr)
        return error.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
