"""Remote GPU benchmark campaign over SSH.

Lifecycle per deployment (laptop drives, GPU host executes)::

    audit host ─► bootstrap venv + wheel ─► pull pinned image (record digest)
      ─► start engine container (127.0.0.1 only) ─► wait for /health
      ─► discover served model id ─► smoke check (1 request + 1 stream)
      ─► run benchmarks ON the host (no WAN in measurements)
      ─► copy lightweight results back ─► stop & remove container ─► delete temp configs

The campaign never creates, resizes or terminates cloud machines, and never copies
credentials: engines read secrets from an env file the user placed on the host.
Without ``--execute`` the CLI drives this class with a dry-run executor and prints
the exact command sequence instead.
"""

from __future__ import annotations

import asyncio
import json
import re
import tempfile
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from inferencelab.backends.launch import LaunchPlan, build_launch_plan
from inferencelab.benchmarking.config import BenchmarkConfig, HTTPTargetConfig
from inferencelab.domain.deployment import BackendKind, DeploymentSpec
from inferencelab.domain.model import ModelSpec
from inferencelab.experiments.plan import ExperimentPlan, PlannedRun
from inferencelab.remote.doctor import gpu_doctor
from inferencelab.remote.executor import CommandExecutor, CommandResult, DryRunExecutor
from inferencelab.remote.host import RemoteHost, expand_remote_path

__all__ = ["CampaignError", "RemoteCampaign", "StepLog", "dry_run_responders"]

_METRICS_PATH = {
    BackendKind.VLLM: "/metrics",
    BackendKind.SGLANG: "/metrics",
    BackendKind.TENSORRT_LLM: "/prometheus/metrics",
}


class CampaignError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class StepLog:
    step: str
    detail: str


Sleep = Callable[[float], Awaitable[None]]


class RemoteCampaign:
    def __init__(
        self,
        plan: ExperimentPlan,
        host: RemoteHost,
        *,
        models: Mapping[str, ModelSpec],
        executor: CommandExecutor,
        local_results: Path,
        wheel: Path | None = None,
        poll_interval_s: float = 10.0,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        if plan.spec.mode != "remote":
            raise ValueError("RemoteCampaign needs a remote experiment plan")
        self.plan = plan
        self.host = host
        self.models = models
        self.executor = executor
        self.local_results = local_results
        self.wheel = wheel
        self.poll_interval_s = poll_interval_s
        self._sleep = sleep
        self.log: list[StepLog] = []

    def _note(self, step: str, detail: str) -> None:
        self.log.append(StepLog(step, detail))

    async def _run(
        self, argv: Sequence[str], *, timeout_s: float = 600.0, check: bool = True
    ) -> CommandResult:
        return await self.executor.run(argv, timeout_s=timeout_s, check=check)

    # ------------------------------------------------------------------ steps
    async def _home(self) -> str:
        result = await self._run(["sh", "-c", 'printf %s "$HOME"'])
        home = result.stdout.strip()
        if not home.startswith("/"):
            raise CampaignError("could not determine the remote home directory")
        return home

    async def _audit(self) -> None:
        checks = await gpu_doctor(self.executor, expect_gpu=True)
        for c in checks:
            self._note("audit", f"{c.name}: {c.status} — {c.detail}")
        failed = [c for c in checks if c.status == "fail"]
        if failed:
            raise CampaignError(
                "host audit failed: " + "; ".join(f"{c.name}: {c.detail}" for c in failed)
            )

    async def _upload_text(self, content: str, remote_dir: str, name: str) -> str:
        with tempfile.TemporaryDirectory() as tmp:
            local = Path(tmp) / name
            await asyncio.to_thread(local.write_text, content, encoding="utf-8")
            await self.executor.put(local, f"{remote_dir}/{name}")
        return f"{remote_dir}/{name}"

    async def _wait_healthy(self, launch: LaunchPlan) -> None:
        url = f"http://127.0.0.1:{launch.port}{launch.health_path}"
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.host.startup_timeout_s
        while True:
            probe = await self._run(
                ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", url],
                check=False,
                timeout_s=30,
            )
            if probe.stdout.strip() == "200":
                self._note("health", f"{launch.deployment} healthy")
                return
            state = await self._run(
                ["docker", "inspect", "-f", "{{.State.Running}}", launch.container_name],
                check=False,
            )
            if state.stdout.strip() == "false":
                logs = await self._run(
                    ["docker", "logs", "--tail", "80", launch.container_name], check=False
                )
                raise CampaignError(
                    f"{launch.deployment} container exited:\n{(logs.stdout + logs.stderr)[-3000:]}"
                )
            if loop.time() > deadline:
                raise CampaignError(
                    f"{launch.deployment} not healthy after {self.host.startup_timeout_s:.0f}s"
                )
            await self._sleep(self.poll_interval_s)

    async def _served_model(self, launch: LaunchPlan) -> str:
        out = await self._run(["curl", "-fsS", f"http://127.0.0.1:{launch.port}/v1/models"])
        try:
            return str(json.loads(out.stdout)["data"][0]["id"])
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise CampaignError(f"unexpected /v1/models response: {out.stdout[:200]}") from exc

    def _bench_config(
        self,
        run: PlannedRun,
        dep: DeploymentSpec,
        model: ModelSpec,
        served: str,
        tags: dict[str, str],
    ) -> BenchmarkConfig:
        base = f"http://127.0.0.1:{dep.port}"
        hardware = self.plan.spec.pricing or dep.hardware
        return BenchmarkConfig(
            name=run.label,
            execution="gpu",
            target=HTTPTargetConfig(
                base_url=base,
                metrics_url=base + _METRICS_PATH.get(dep.backend, "/metrics"),
                engine=dep.backend.value if dep.backend.is_gpu_engine else None,  # type: ignore[arg-type]
            ),
            workload=run.workload.model_copy(update={"model": served}),
            slo=self.plan.spec.slo,
            deployment=dep.model_copy(update={"hardware": hardware}),
            model=model,
            simulated=False,
            tags=tags,
        )

    async def run(self) -> list[Path]:
        spec = self.plan.spec
        home = await self._home()
        work = expand_remote_path(self.host.workdir, home)
        hf_cache = expand_remote_path(self.host.hf_cache, home)
        env_file = expand_remote_path(self.host.env_file, home)
        venv_bin = f"{work}/.venv/bin"
        await self._audit()
        await self._run(["mkdir", "-p", f"{work}/cfg", f"{work}/results", f"{work}/dist"])
        has_env = (await self._run(["test", "-f", env_file], check=False)).ok
        if not has_env:
            self._note("bootstrap", f"{env_file} missing: gated models need HF_TOKEN there")
        if self.wheel is not None:
            await self.executor.put(self.wheel, f"{work}/dist/{self.wheel.name}")
            await self._run(["uv", "venv", "--allow-existing", "--python", "3.12", f"{work}/.venv"])
            await self._run(
                [
                    "uv",
                    "pip",
                    "install",
                    "--python",
                    f"{work}/.venv/bin/python",
                    f"{work}/dist/{self.wheel.name}",
                ],
                timeout_s=1800,
            )
        collected: list[Path] = []
        grouped: dict[str, list[PlannedRun]] = {}
        for run in self.plan.runs:
            grouped.setdefault(run.target, []).append(run)
        for dep_name, runs in grouped.items():
            dep = self.plan.deployments[dep_name]
            model = self.models.get(dep.model)
            if model is None:
                raise CampaignError(
                    f"model {dep.model!r} for deployment {dep_name!r} is not registered"
                )
            collected += await self._deployment(
                dep,
                model,
                runs,
                work,
                hf_cache,
                env_file if has_env else None,
                venv_bin,
                spec.max_run_duration_s,
            )
        return collected

    async def _deployment(
        self,
        dep: DeploymentSpec,
        model: ModelSpec,
        runs: list[PlannedRun],
        work: str,
        hf_cache: str,
        env_file: str | None,
        venv_bin: str,
        max_run_s: float,
    ) -> list[Path]:
        launch = build_launch_plan(dep, model)
        cfg_dir = f"{work}/cfg/{dep.name}"
        await self._run(["mkdir", "-p", cfg_dir])
        for name, content in launch.config_files.items():
            await self._upload_text(content, cfg_dir, name)
        for cmd in launch.prepare:
            self._note("prepare", " ".join(cmd))
            await self._run(cmd, timeout_s=7200)
        await self._run(["docker", "pull", launch.image], timeout_s=3600)
        digest = await self._run(
            ["docker", "image", "inspect", "--format", "{{index .RepoDigests 0}}", launch.image],
            check=False,
        )
        gpu = await self._run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], check=False
        )
        tags = {
            "image_digest": digest.stdout.strip(),
            "gpu_driver": gpu.stdout.strip().splitlines()[0] if gpu.stdout.strip() else "",
            "deployment": dep.name,
            "host": self.host.name,
        }
        await self._run(["docker", "rm", "-f", launch.container_name], check=False)
        collected: list[Path] = []
        try:
            await self._run(
                launch.docker_run(hf_cache=hf_cache, env_file=env_file, config_dir=cfg_dir)
            )
            self._note("start", f"{dep.name}: {launch.image}")
            await self._wait_healthy(launch)
            served = await self._served_model(launch)
            backend_cfg = {
                "type": dep.backend.value,
                "name": dep.name,
                "base_url": f"http://127.0.0.1:{dep.port}",
                "models": [served],
                "metrics_path": _METRICS_PATH.get(dep.backend, "/metrics"),
            }
            check_path = await self._upload_text(
                yaml.safe_dump(backend_cfg), cfg_dir, "backend.yaml"
            )
            await self._run(
                [f"{venv_bin}/inferencelab", "backend", "check", check_path], timeout_s=600
            )
            for i, run in enumerate(runs):
                bench = self._bench_config(run, dep, model, served, tags)
                path = await self._upload_text(
                    yaml.safe_dump(
                        bench.model_dump(mode="json", exclude_none=True), sort_keys=False
                    ),
                    cfg_dir,
                    f"bench-{i:03d}.yaml",
                )
                out = await self._run(
                    [
                        f"{venv_bin}/inferencelab",
                        "benchmark",
                        "run",
                        path,
                        "--execute",
                        "--output",
                        f"{work}/results",
                    ],
                    timeout_s=max_run_s + 900,
                )
                match = re.search(r"^run_dir=(\S+)$", out.stdout, re.MULTILINE)
                if match is None:
                    raise CampaignError(
                        f"benchmark did not report its run directory:\n{out.stdout[-500:]}"
                    )
                dest = self.local_results / self.host.name / dep.name
                await self.executor.get(match.group(1), dest)
                collected.append(dest / Path(match.group(1)).name)
                self._note("collect", f"{run.label} → {dest}")
        finally:
            await self._run(
                ["docker", "stop", "-t", "30", launch.container_name], check=False, timeout_s=120
            )
            await self._run(["docker", "rm", launch.container_name], check=False)
            await self._run(["rm", "-rf", cfg_dir], check=False)
            self._note(
                "cleanup",
                f"{dep.name}: container removed, temp configs deleted (machine left running)",
            )
        return collected


def dry_run_responders(home: str = "/home/ubuntu") -> DryRunExecutor:
    """A dry-run executor with plausible outputs, so `remote plan` can print every step."""

    def respond(argv: tuple[str, ...]) -> CommandResult | None:
        joined = " ".join(argv)
        if argv[:2] == ("sh", "-c") and "HOME" in joined:
            return CommandResult(argv, 0, home, "", 0.0)
        if argv[:2] == ("sh", "-c") and "nvidia-smi" in joined:
            return CommandResult(argv, 0, "/usr/bin/nvidia-smi", "", 0.0)
        if argv[0] == "nvidia-smi" and any("--query-gpu=name,memory.total" in a for a in argv):
            return CommandResult(argv, 0, "<GPU>, 81559, <driver>", "", 0.0)
        if argv[0] == "nvidia-smi" and len(argv) == 1:
            return CommandResult(argv, 0, "CUDA Version: <cuda>", "", 0.0)
        if argv[:2] == ("docker", "version"):
            return CommandResult(argv, 0, "<docker>", "", 0.0)
        if argv[:2] == ("docker", "info"):
            return CommandResult(argv, 0, '{"nvidia":{}}', "", 0.0)
        if argv[:2] == ("df", "-Pk"):
            return CommandResult(
                argv, 0, "Filesystem 1K Used Avail\n/dev/x 1 1 900000000 1% /", "", 0.0
            )
        if argv[0] == "curl" and argv[-1].endswith("/v1/models"):
            return CommandResult(argv, 0, '{"data":[{"id":"<served-model>"}]}', "", 0.0)
        if argv[0] == "curl":
            return CommandResult(argv, 0, "200", "", 0.0)
        if len(argv) > 2 and argv[1:3] == ("benchmark", "run"):
            return CommandResult(
                argv, 0, f"run_dir={home}/inferencelab/results/<run-id>\n", "", 0.0
            )
        return None

    return DryRunExecutor(responders=[respond])
