from __future__ import annotations

import shlex
from pathlib import Path

import pydantic
import pytest

from inferencelab.config.loader import load_model_file
from inferencelab.domain.model import ModelSpec
from inferencelab.experiments.plan import expand_experiment
from inferencelab.experiments.spec import ExperimentSpec
from inferencelab.remote.campaign import CampaignError, RemoteCampaign, dry_run_responders
from inferencelab.remote.doctor import gpu_doctor, parse_cuda_version, parse_gpu_csv
from inferencelab.remote.executor import CommandError, CommandResult, DryRunExecutor, SSHExecutor
from inferencelab.remote.host import RemoteHost, expand_remote_path
from tests.conftest import CONFIGS


def host(**kw: object) -> RemoteHost:
    base: dict[str, object] = {"name": "gpu", "hostname": "gpu.example.com", "user": "ubuntu"}
    base.update(kw)
    return RemoteHost.model_validate(base)


def test_host_validation_blocks_injection() -> None:
    for bad in (
        {"hostname": "a;rm -rf /"},
        {"user": "root$(id)"},
        {"ssh_options": ["ProxyCommand=evil"]},
    ):
        with pytest.raises(pydantic.ValidationError):
            host(**bad)
    assert host().target == "ubuntu@gpu.example.com"
    assert expand_remote_path("~/x", "/home/u") == "/home/u/x"
    assert (
        expand_remote_path("~", "/home/u") == "/home/u"
        and expand_remote_path("/abs", "/h") == "/abs"
    )


def test_ssh_argv_quotes_the_remote_command() -> None:
    ssh = SSHExecutor(
        host(port=2222, identity_file="/keys/id", ssh_options=["ServerAliveCountMax=3"])
    )
    argv = ssh.ssh_argv(["echo", "a b; touch /tmp/pwned"])
    assert argv[0] == "ssh" and "BatchMode=yes" in argv and "-p" in argv and "2222" in argv
    assert argv[argv.index("-i") + 1] == "/keys/id"
    assert argv[-2] == "--" and shlex.split(argv[-1]) == ["echo", "a b; touch /tmp/pwned"]


def test_gpu_output_parsers() -> None:
    gpus = parse_gpu_csv("NVIDIA H100 80GB HBM3, 81559, 580.65\nbad line\n")
    assert len(gpus) == 1 and gpus[0].memory_mib == 81559 and gpus[0].driver == "580.65"
    assert (
        parse_cuda_version("| NVIDIA-SMI 580.65  Driver Version: 580.65  CUDA Version: 13.0 |")
        == "13.0"
    )
    assert parse_cuda_version("nothing") is None


async def test_gpu_doctor_on_cpu_laptop_skips_gpu() -> None:
    ex = DryRunExecutor(responders=[lambda argv: CommandResult(argv, 1, "", "missing", 0.0)])
    checks = await gpu_doctor(ex, expect_gpu=False)
    by_name = {c.name: c.status for c in checks}
    assert by_name["nvidia-smi"] == "skip" and by_name["docker"] == "warn"


async def test_gpu_doctor_on_gpu_host_with_fake_outputs() -> None:
    ex = dry_run_responders()
    checks = await gpu_doctor(
        ex, expect_gpu=True, container_test_image="nvidia/cuda:13.0.0-base-ubuntu24.04"
    )
    by_name = {c.name: c.status for c in checks}
    assert (
        by_name["gpus"] == "ok"
        and by_name["docker"] == "ok"
        and by_name["nvidia-container-runtime"] == "ok"
    )
    assert by_name["disk"] == "ok" and "container-gpu" in by_name


async def test_dry_run_executor_scripted_failure() -> None:
    ex = DryRunExecutor(responders=[lambda argv: CommandResult(argv, 2, "", "boom", 0.0)])
    with pytest.raises(CommandError, match="boom"):
        await ex.run(["false"])
    assert (await ex.run(["false"], check=False)).returncode == 2


def _campaign(tmp_path: Path, executor: DryRunExecutor) -> RemoteCampaign:
    spec = load_model_file(CONFIGS / "experiments" / "gpu_first_comparison.yaml", ExperimentSpec)
    narrowed = spec.model_copy(
        update={
            "matrix": spec.matrix.model_copy(
                update={"workloads": spec.matrix.workloads[:1], "concurrency": [1]}
            )
        }
    )
    plan = expand_experiment(narrowed, CONFIGS / "experiments")
    models = {
        m.name: m
        for m in (load_model_file(p, ModelSpec) for p in (CONFIGS / "models").glob("*.yaml"))
    }
    return RemoteCampaign(
        plan,
        host(),
        models=models,
        executor=executor,
        local_results=tmp_path,
        wheel=tmp_path / "x.whl",
        poll_interval_s=0.0,
    )


async def test_campaign_dry_run_lifecycle(tmp_path: Path) -> None:
    ex = dry_run_responders()
    campaign = _campaign(tmp_path, ex)
    collected = await campaign.run()
    joined = [" ".join(c) for c in ex.commands]
    pulls = [c for c in joined if c.startswith("docker pull")]
    runs = [c for c in ex.commands if c[:2] == ("docker", "run")]
    assert len(pulls) == 2 and len(runs) == 2  # vLLM + SGLang
    assert all("127.0.0.1:" in " ".join(r) for r in runs)  # never published publicly
    assert all(
        any(a.endswith("@sha256:" + a.rsplit("@sha256:", 1)[-1]) for a in r if "@sha256:" in a)
        for r in runs
    )
    assert sum("benchmark run" in c and "--execute" in c for c in joined) == 2
    assert any(c.startswith("docker stop") for c in joined) and any(
        c.startswith("rm -rf") for c in joined
    )
    assert not any(w in c for c in joined for w in ("shutdown", "poweroff", "terminate"))
    assert len(collected) == 2 and any(t[0] == "get" for t in ex.transfers)
    assert any(step.step == "cleanup" for step in campaign.log)


async def test_campaign_aborts_when_audit_fails(tmp_path: Path) -> None:
    def no_gpu(argv: tuple[str, ...]) -> CommandResult | None:
        if argv[:2] == ("sh", "-c") and "HOME" in " ".join(argv):
            return CommandResult(argv, 0, "/home/u", "", 0.0)
        return CommandResult(argv, 1, "", "missing", 0.0)

    with pytest.raises(CampaignError, match="audit failed"):
        await _campaign(tmp_path, DryRunExecutor(responders=[no_gpu])).run()


async def test_campaign_reports_container_crash(tmp_path: Path) -> None:
    base = dry_run_responders()

    def crashed(argv: tuple[str, ...]) -> CommandResult | None:
        if argv[0] == "curl" and not argv[-1].endswith("/v1/models"):
            return CommandResult(argv, 0, "000", "", 0.0)
        if argv[:2] == ("docker", "inspect"):
            return CommandResult(argv, 0, "false", "", 0.0)
        return None

    ex = DryRunExecutor(responders=[crashed, *base.responders])
    with pytest.raises(CampaignError, match="exited"):
        await _campaign(tmp_path, ex).run()
    assert any(c[:2] == ("docker", "stop") for c in ex.commands)  # cleanup still ran
