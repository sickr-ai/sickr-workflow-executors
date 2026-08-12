"""Shared executor-step runner for governed hooks and worker nodes.

The hook/worker customization model has two distinct extension points:

* the runner implementation, such as ``builtin://runtime/hook_runner.py``;
* the executor registry, a library of small checks/actions composed by state
  config.

This module owns the built-in registry and the common aggregation rules. Custom
runner code may implement the same input/output contract, but SICKR defaults
should stay data-driven through these executor steps.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping, Protocol, Sequence
from urllib.parse import urlparse

from labudi_orchestrator.agent_workflow_client import AgentWorkflowClientError
from labudi_orchestrator.completion_actions import (
    authorize_completion,
    completion_audit_record,
)
from sickr_workflow_executors.executor_manifest import (
    BUILTIN_EXECUTOR_MANIFESTS,
    assert_registration_conformance,
    validate_step_executor_reference,
)
from labudi_orchestrator.github_credential_environment import (
    GitHubAccess,
    GitHubCredentialEnvironmentError,
    resolve_github_credential_environment,
)
from labudi_orchestrator.github_pr_checks import read_commit_checks
from labudi_orchestrator.obligation_executors import ExecutorContext
from labudi_orchestrator.proc import run_process
from labudi_orchestrator.ticket_branch_publication import github_repo_from_remote, github_token_env, publish_ticket_branch
from labudi_orchestrator.workflow_code_source import SourceResolver, WorkflowCodeSourceError


StepStatus = Literal["passed", "failed", "error", "skipped"]
# Per-step failure routing intent (design 2026-07-09). Legacy `{warn, failure}`
# normalize to `{continue, retry}` on read (behavior-preserving).
OnFailure = Literal["continue", "retry", "escalate", "fail", "error"]
_LEGACY_ON_FAILURE = {"warn": "continue", "failure": "retry"}
_VALID_ON_FAILURE = {"continue", "retry", "escalate", "fail", "error"}
_STEP_ID_ALLOWED_RE = re.compile(r"[^A-Za-z0-9._:-]")
_PR_URL_RE = re.compile(r"https://github\.com/(?P<owner>[^/\s]+)/(?P<repo>[^/\s]+)/pull/(?P<number>\d+)")
_SECRET_CONTENT_RE = re.compile(r"(?:-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|gh[opsu]_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16})")


def _is_prohibited_commit_path(path: str) -> bool:
    item = Path(path)
    return item.name.casefold() in {".env", "agent.env", "credentials.json", "id_rsa", "id_ed25519"} or item.suffix.casefold() in {".pem", ".p12", ".pfx", ".key"}


def _normalize_on_failure(raw: Any) -> str:
    """Map an authored `on_failure` to the canonical vocabulary. Legacy values
    (`warn`/`failure`) and missing/invalid values normalize to preserve today's
    behavior: `warn`->`continue` (non-blocking), everything else->`retry`."""
    if isinstance(raw, str):
        mapped = _LEGACY_ON_FAILURE.get(raw, raw)
        if mapped in _VALID_ON_FAILURE:
            return mapped
    return "retry"


def step_failure_disposition(on_failure: str) -> Literal["retry", "terminal"] | None:
    """Retry intent carried across the orch->workflow-service boundary for a
    `failure` outcome. `retry`->retry to the state cap; `escalate`/`fail`->
    terminal (no retry, route the fail branch). `continue`->skip and `error`->
    error are distinct outcomes and carry no disposition."""
    if on_failure == "retry":
        return "retry"
    if on_failure in {"escalate", "fail"}:
        return "terminal"
    return None


def _sanitize_step_id(step_id: str) -> str:
    """Constrain a step id before it enters the operator-facing failure reason."""
    return _STEP_ID_ALLOWED_RE.sub("", step_id)


@dataclass(frozen=True)
class ExecutorStep:
    executor_id: str
    executor_contract_version: Any = None
    enabled: bool = True
    required: bool = True
    on_failure: OnFailure = "retry"
    params: dict[str, Any] = field(default_factory=dict)
    source: dict[str, Any] | None = None
    timeout_seconds: int | None = None
    # What the executor declared it needs, stamped into the contract at save time
    # from the version this step pins. Carried WITH the step, exactly like
    # `source`, because the runtime cannot read D1: an org executor's declaration
    # lives there and would otherwise be invisible here.
    #
    # None means undeclared, which is not "needs nothing".
    requires: dict[str, Any] | None = None


@dataclass(frozen=True)
class ExecutorStepResult:
    executor_id: str
    status: StepStatus
    message: str
    evidence: dict[str, Any] = field(default_factory=dict)
    # Distinguishes a fault in the environment from a fault in the workflow.
    # Only WORKFLOW_SETUP_ERROR today; see SETUP_ERROR_BLOCKER.
    blocker_kind: str | None = None

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "executor_id": self.executor_id,
            "status": self.status,
            "message": self.message,
            "evidence": self.evidence,
        }
        if self.blocker_kind is not None:
            out["blocker_kind"] = self.blocker_kind
        return out


@dataclass(frozen=True)
class DeployPollResult:
    """Outcome of waiting for a deploy: what was expected, what was served."""

    matched: bool
    deployed: str | None
    expected: str
    attempts: int
    last_error: str | None


@dataclass(frozen=True)
class GitStepResult:
    returncode: int
    stdout: str
    stderr: str

    def evidence(self, prefix: str) -> dict[str, Any]:
        return {
            f"{prefix}_exit_code": self.returncode,
            f"{prefix}_stdout": _truncate(self.stdout, 8192),
            f"{prefix}_stderr": _truncate(self.stderr, 8192),
        }


class StepExecutor(Protocol):
    def __call__(self, *, step: ExecutorStep, ctx: ExecutorContext, actor_result: dict[str, Any] | None) -> ExecutorStepResult: ...


SETUP_ERROR_BLOCKER = "WORKFLOW_SETUP_ERROR"

# Three-valued verification (Contract B). A verifier that could not observe
# the world — network fault, rate limit, credential outage — is INCONCLUSIVE:
# it proved neither success nor failure, so it must not blame the
# implementation and must not trigger a rerun of Main. It routes as a system
# fault (status "error") carrying this blocker, which the retry path resumes
# with preserved main evidence — postflight re-verification only.
VERIFICATION_INCONCLUSIVE_BLOCKER = "VERIFICATION_INCONCLUSIVE"


def _inconclusive(
    executor_id: str, message: str, evidence: dict[str, Any] | None = None
) -> ExecutorStepResult:
    return ExecutorStepResult(
        executor_id,
        "error",
        message,
        {**(evidence or {}), "verification": "inconclusive"},
        blocker_kind=VERIFICATION_INCONCLUSIVE_BLOCKER,
    )

# shutil.which walks PATH on the filesystem. Steps repeat within a phase and
# phases repeat across tickets, so a positive answer is cached for the life of
# the process. A negative one never is — see _tool_present.
_TOOL_PRESENT_CACHE: set[str] = set()


def _tool_present(tool: str) -> bool:
    """Only PRESENT tools are cached, and deliberately so.

    Caching a miss means an operator who reads "gh is not installed on this
    runner", installs gh, and retries the ticket is refused again — by a cached
    answer, with the tool sitting right there. The gate would be telling them to
    do the thing they had just done, and the only remedy would be restarting the
    daemon, which nothing in the message suggests.

    A present tool does not become absent mid-run in any way worth defending
    against, so caching that direction costs nothing. A miss re-probes: PATH is
    walked again, which is the expensive case, but it happens once per refused
    step and a refused step does no other work.
    """
    if tool in _TOOL_PRESENT_CACHE:
        return True
    if shutil.which(tool) is None:
        return False
    _TOOL_PRESENT_CACHE.add(tool)
    return True


def _declared_requirements(step: ExecutorStep) -> Mapping[str, Any] | None:
    """What this step's executor says it needs, or None when undeclared.

    The CONTRACT wins. A published step is stamped at save time with the
    requirements of the version it pins, and that is the authoritative answer:
    it is the only way an org executor's declaration — which lives in D1, where
    the runtime cannot look — reaches this process at all. It is also pinned,
    where the local registry is whatever this daemon happens to ship.

    The built-in registry is the fallback, for contracts published before
    stamping existed. Without it, every step in an older workflow would go from
    gated to undeclared on deploy.

    None is NOT "needs nothing". An org executor whose author declared nothing,
    and the main-phase operations the observation harness does not drive, both
    land here — and gating them as verified-clean would assert something nobody
    has checked.
    """
    if step.requires is not None:
        return step.requires
    descriptor = BUILTIN_EXECUTOR_MANIFESTS.get(step.executor_id)
    return None if descriptor is None else descriptor.requires


def unmet_step_requirements(step: ExecutorStep, ctx: ExecutorContext) -> ExecutorStepResult | None:
    """Refuse a step whose declared requirements this runner cannot satisfy.

    Returns None when the step may run — including when it declares nothing,
    which means unknown rather than satisfied.

    This is a setup fault, not a policy verdict: the executor never ran, so it
    decided nothing. It routes as `error` (a system fault) carrying a distinct
    blocker, so an operator can tell "this machine is missing gh" from "this
    executor is broken" — different problems, fixed by different people.

    TOOLS ONLY, deliberately. A declared `connections` requirement is NOT gated,
    because an absent connection does not stop a step running:
    resolve_github_credential_environment falls back to `local_profile` whenever
    there is no lease or no brokered method, so the step proceeds on the
    machine's own git/gh credentials. Refusing there would turn a documented
    fallback into a blocked ticket. `connections` stays descriptive — worth
    showing an operator what a state reaches for — not prescriptive.

    Known limit, stated because it is easy to over-trust: this is a PRESENCE
    check. `gh` installed but unauthenticated passes it and still fails inside
    the step, which is exactly how the original production failure looked.
    Catching that needs an auth probe — a different and more expensive check.
    """
    requires = _declared_requirements(step)
    if requires is None:
        return None

    missing_tools = sorted(t for t in requires.get("tools", ()) if not _tool_present(t))
    if not missing_tools:
        return None

    listed = ", ".join(missing_tools)
    reason = (
        f"{listed} are not installed on this runner"
        if len(missing_tools) > 1
        else f"{listed} is not installed on this runner"
    )

    return ExecutorStepResult(
        step.executor_id,
        "error",
        f"{step.executor_id} cannot run: {reason}",
        # Names only, and only names that came from the manifest. No PATH
        # contents, no environment, no credential shape — nothing that varies
        # with the host and nothing that was read off it.
        {
            "setup": {
                "missing_tools": missing_tools,
                "tools_verified": "presence_only",
            },
        },
        blocker_kind=SETUP_ERROR_BLOCKER,
    )


# Blockers that must survive the hook aggregation boundary. Anything not in
# this set falls back to WORKFLOW_PROTOCOL_ERROR at the hook layer — which is
# exactly what must NOT happen to an inconclusive verification, or a verifier
# outage turns back into a protocol fault and the resumable-postflight routing
# never sees it.
_FORWARDED_PHASE_BLOCKERS = (SETUP_ERROR_BLOCKER, VERIFICATION_INCONCLUSIVE_BLOCKER)


def phase_blocker_kind(results: Iterable[ExecutorStepResult]) -> str | None:
    """The phase-level blocker to surface, if any step reported a known one.

    First match wins in result order — the deciding step is the first
    non-pass, and its blocker is the phase's story.
    """
    for result in results:
        if result.blocker_kind in _FORWARDED_PHASE_BLOCKERS:
            return result.blocker_kind
    return None


def setup_blocker_kind(results: Iterable[ExecutorStepResult]) -> str | None:
    """The setup blocker to surface for a phase, if any step reported one."""
    for result in results:
        if result.blocker_kind == SETUP_ERROR_BLOCKER:
            return SETUP_ERROR_BLOCKER
    return None


def progress_step_view(
    steps: Sequence[ExecutorStep],
    results: Sequence[ExecutorStepResult],
    *,
    running: str | None = None,
) -> list[dict[str, str]]:
    """The per-step view published while a phase runs.

    Three fields only — executor_id, status, message. Step evidence carries
    command output, environment, and file contents; none of it belongs in a
    record whose purpose is "which step is running, and why did one fail".
    The service sanitises this again on arrival, so this is the first of two
    gates rather than the only one.

    Steps that have not been reached yet are included as `pending` rather than
    omitted. A list that grows as it executes cannot be told apart from a phase
    that only ever had the steps completed so far, and the whole point is to say
    up front what is going to run.
    """
    view: list[dict[str, str]] = []
    done = {result.executor_id for result in results}
    for result in results:
        view.append({
            "executor_id": result.executor_id,
            "status": result.status,
            "message": result.message,
        })
    for step in steps:
        if step.executor_id in done:
            continue
        if step.executor_id == running:
            view.append({"executor_id": step.executor_id, "status": "running", "message": ""})
        else:
            view.append({"executor_id": step.executor_id, "status": "pending", "message": ""})
    return view


def run_executor_steps(
    *,
    config_json: dict[str, Any] | None,
    ctx: ExecutorContext,
    actor_result: dict[str, Any] | None = None,
    registry: dict[str, StepExecutor] | None = None,
    on_progress: Callable[[list[dict[str, str]]], None] | None = None,
) -> tuple[Literal["passed", "failure", "error", "system_error", "skipped"], list[ExecutorStepResult]]:
    steps = parse_executor_steps(config_json)
    executors = registry or DEFAULT_STEP_EXECUTORS
    results: list[ExecutorStepResult] = []

    def publish(running: str | None) -> None:
        """Report progress without ever being able to affect the phase.

        Visibility is strictly less important than the work it describes, so
        every failure here — a dead network, a 409 from a settled run, a bug in
        the callback — is swallowed. A state must never fail because we could
        not say what it was doing.
        """
        if on_progress is None:
            return
        try:
            on_progress(progress_step_view(steps, results, running=running))
        except Exception:  # noqa: BLE001 - progress reporting is never fatal
            pass

    for step in steps:
        if not step.enabled:
            results.append(ExecutorStepResult(step.executor_id, "skipped", "step disabled"))
            continue
        # Published at the TOP of the iteration, which reports two things at
        # once: this step is starting, and every earlier step's settled result.
        # One call site per loop rather than one beside each of the many
        # `results.append(...)` paths below — those are easy to add and easy to
        # forget, and a missed one shows a step as running after it finished.
        publish(step.executor_id)
        reference_error = validate_step_executor_reference(
            executor_id=step.executor_id,
            executor_contract_version=step.executor_contract_version,
            source=step.source,
        )
        if reference_error is not None:
            results.append(
                ExecutorStepResult(
                    step.executor_id,
                    "error",
                    reference_error.reason,
                    reference_error.as_evidence(workflow_contract_version=None),
                )
            )
            break
        step_ctx = replace(ctx, prior_step_results=tuple(result.as_dict() for result in results))
        # The requirements gate sits HERE, before dispatch, rather than as a
        # decorator on each executor. Every path below — registered builtin,
        # org-registered script, version-skewed builtin — goes through this
        # point, so a new executor cannot be added without it, and no author has
        # to remember to apply anything.
        setup_error = unmet_step_requirements(step, step_ctx)
        if setup_error is not None:
            results.append(setup_error)
            if _step_result_blocks_following_steps(step, setup_error):
                break
            continue
        src_kind = step.source.get("source") if isinstance(step.source, dict) else None
        # Source identity is part of the pinned step contract. An organization
        # or team installation may intentionally reuse a SICKR logical id; in
        # that case its pinned repository implementation must win over the
        # bundled handler. Falling back by id here would make overrides appear
        # valid in the catalog while silently executing different code.
        executor = None if src_kind == "org_repo" else executors.get(step.executor_id)
        if executor is None:
            if src_kind in {"org_repo", "sickr_default"}:
                # A materializable external/script source — resolve + run it.
                results.append(_run_script_step_executor(step=step, ctx=step_ctx, actor_result=actor_result))
                if _step_result_blocks_following_steps(step, results[-1]):
                    break
            elif src_kind == "sickr_builtin":
                # A KNOWN builtin (declares a sickr_builtin source) that is not in
                # THIS runtime's registry — a version skew, not a system fault. Do
                # NOT mis-route it through the script materializer (which would reject
                # `sickr_builtin` with a cryptic code_source error). Route as a
                # business "failed" so the step's on_failure edge decides (retriable
                # by default); it will run for real once the daemon is updated.
                results.append(ExecutorStepResult(
                    step.executor_id,
                    "failed",
                    f"executor {step.executor_id!r} is not registered in this runtime "
                    f"(the daemon may be outdated)",
                ))
                if _step_result_blocks_following_steps(step, results[-1]):
                    break
            else:
                # No source / unknown source kind: a genuine config error (unknown
                # executor id). This stays a system fault.
                results.append(ExecutorStepResult(
                    step.executor_id,
                    "error",
                    f"executor {step.executor_id!r} is not registered",
                ))
                if _step_result_blocks_following_steps(step, results[-1]):
                    break
            continue
        try:
            results.append(_run_builtin_step_executor(executor=executor, step=step, ctx=step_ctx, actor_result=actor_result))
        except Exception as err:  # noqa: BLE001 - executor failure is evidence
            results.append(ExecutorStepResult(
                step.executor_id,
                "error",
                f"executor raised: {err!r}",
            ))
        if _step_result_blocks_following_steps(step, results[-1]):
            break
    # The final settled view. Without this the last step would stay "running"
    # until evidence lands, and a phase that broke out early would never publish
    # the result that stopped it.
    publish(None)
    outcome, _failing = resolve_step_outcome(steps, results)
    return outcome, results


def _step_result_blocks_following_steps(step: ExecutorStep, result: ExecutorStepResult) -> bool:
    if result.status in {"passed", "skipped"}:
        return False
    if not step.required or step.on_failure == "continue":
        return False
    return True


def parse_executor_steps(config_json: dict[str, Any] | None) -> list[ExecutorStep]:
    if not isinstance(config_json, dict):
        return []
    raw_steps = config_json.get("steps")
    if not isinstance(raw_steps, list):
        return []
    steps: list[ExecutorStep] = []
    for raw in raw_steps:
        if not isinstance(raw, dict):
            continue
        executor_id = str(raw.get("executor_id") or "").strip()
        if not executor_id:
            continue
        params = raw.get("params") if isinstance(raw.get("params"), dict) else {}
        on_failure = _normalize_on_failure(raw.get("on_failure"))
        steps.append(ExecutorStep(
            executor_id=executor_id,
            executor_contract_version=raw.get("executor_contract_version"),
            enabled=raw.get("enabled") is not False,
            required=raw.get("required") is not False,
            on_failure=on_failure,  # type: ignore[arg-type]
            params=dict(params),
            source=raw.get("source") if isinstance(raw.get("source"), dict) else None,
            requires=raw.get("requires") if isinstance(raw.get("requires"), dict) else None,
            timeout_seconds=_bounded_timeout(raw.get("timeout_seconds")),
        ))
    return steps


def resolve_step_outcome(
    steps: list[ExecutorStep],
    results: list[ExecutorStepResult],
) -> tuple[Literal["passed", "failure", "error", "system_error", "skipped"], ExecutorStepResult | None]:
    """Sequential, no aggregation. The first non-pass step decides the phase
    outcome and stops. A status=="error" result is a SYSTEM fault (executor
    missing / raised / malformed) and maps to system_error; a status=="failed"
    result is a business failure routed by the step's on_failure edge:
    ``error`` -> error; ``retry``/``escalate``/``fail`` -> failure (the
    retry-vs-terminal distinction is carried via ``step_failure_disposition``);
    ``continue`` is non-blocking.

    ``required=False`` means the step is opportunistic: it still runs and
    records evidence, but failed/error results do not block the state. Use
    ``enabled=False`` when a configured step should not run at all.
    """
    steps_by_id = {step.executor_id: step for step in steps}
    saw_skipped = False
    for result in results:
        if result.status == "skipped":
            step = steps_by_id.get(result.executor_id)
            if step is not None and step.enabled and step.required and step.on_failure != "continue":
                saw_skipped = True
            continue
        if result.status in {"passed", "skipped"}:
            continue
        step = steps_by_id.get(result.executor_id)
        if step is None:
            return "system_error", result
        if not step.required or step.on_failure == "continue":
            continue
        if result.status == "error":
            return "system_error", result
        if step.on_failure == "error":
            return "error", result
        return "failure", result
    return ("skipped" if saw_skipped else "passed"), None


def failure_disposition_fields(
    config_json: dict[str, Any] | None,
    phase: str,
    outcome: str,
    step_results: list[ExecutorStepResult],
) -> dict[str, Any]:
    """Derive the retry-disposition + structured failure-reason fields to merge
    into a hook's ``output`` evidence. Empty unless ``outcome == "failure"``.

    Re-parses ``config_json`` and re-resolves the deciding step (deterministic,
    side-effect-free) so callers keep the existing ``run_executor_steps``
    2-tuple. The deciding step's ``on_failure`` drives ``retry_disposition``
    (missing disposition -> ``"retry"``); ``escalate``/``fail`` also emit the
    structured, sanitized failure reason for the operator."""
    if outcome != "failure":
        return {}
    steps = parse_executor_steps(config_json)
    _outcome, deciding_result = resolve_step_outcome(steps, step_results)
    if deciding_result is None:
        return {}
    deciding = {step.executor_id: step for step in steps}.get(deciding_result.executor_id)
    if deciding is None:
        return {}
    disposition = step_failure_disposition(deciding.on_failure) or "retry"
    fields: dict[str, Any] = {
        "retry_disposition": disposition,
        "on_failure": deciding.on_failure,
        "deciding_step_id": deciding.executor_id,
    }
    if disposition == "terminal":
        step_id = _sanitize_step_id(deciding.executor_id)
        fields["failure_phase"] = phase
        fields["failure_step_id"] = step_id
        fields["failure_reason"] = f"{phase}_step_failed:{step_id}"
    return fields


def _ticket_has_description(*, step: ExecutorStep, ctx: ExecutorContext, actor_result: dict[str, Any] | None) -> ExecutorStepResult:
    description = str(ctx.ticket.get("description") or "").strip()
    if description:
        return ExecutorStepResult(step.executor_id, "passed", "ticket description is present", {"length": len(description)})
    return ExecutorStepResult(step.executor_id, "failed", "ticket description is missing")


def _ticket_has_acceptance_criteria(*, step: ExecutorStep, ctx: ExecutorContext, actor_result: dict[str, Any] | None) -> ExecutorStepResult:
    criteria = ctx.ticket.get("acceptance_criteria")
    if isinstance(criteria, list) and any(str(item).strip() for item in criteria):
        return ExecutorStepResult(step.executor_id, "passed", "acceptance criteria are present", {"count": len(criteria)})
    return ExecutorStepResult(step.executor_id, "failed", "acceptance criteria are missing")


def _ticket_has_workflow(*, step: ExecutorStep, ctx: ExecutorContext, actor_result: dict[str, Any] | None) -> ExecutorStepResult:
    workflow_fields = {
        "type": ctx.ticket.get("type"),
        "workflow_template_id": ctx.ticket.get("workflow_template_id"),
        "workflow_node_id": ctx.ticket.get("workflow_node_id"),
    }
    ticket_type = workflow_fields["type"]
    has_workflow_type = isinstance(ticket_type, str) and ticket_type.startswith("workflow_")
    has_template = _nonempty_str(workflow_fields["workflow_template_id"])
    has_node = _nonempty_str(workflow_fields["workflow_node_id"])
    if has_workflow_type or has_template or has_node:
        return ExecutorStepResult(step.executor_id, "passed", "ticket workflow reference is present", workflow_fields)
    return ExecutorStepResult(step.executor_id, "failed", "ticket workflow reference is missing", workflow_fields)


def _ticket_has_repo(*, step: ExecutorStep, ctx: ExecutorContext, actor_result: dict[str, Any] | None) -> ExecutorStepResult:
    repo = ctx.ticket.get("repo")
    repo_url = ctx.ticket.get("repo_url")
    if _nonempty_str(repo) or _nonempty_str(repo_url):
        return ExecutorStepResult(step.executor_id, "passed", "ticket repository is present", {"repo": repo, "repo_url": repo_url})
    return ExecutorStepResult(step.executor_id, "failed", "ticket repository is missing", {"repo": repo, "repo_url": repo_url})


def _ticket_has_branch(*, step: ExecutorStep, ctx: ExecutorContext, actor_result: dict[str, Any] | None) -> ExecutorStepResult:
    for key in ("branch", "ticket_branch", "head_branch", "base_branch"):
        value = ctx.ticket.get(key)
        if _nonempty_str(value):
            return ExecutorStepResult(step.executor_id, "passed", "ticket branch is present", {"field": key, "branch": value})
    return ExecutorStepResult(step.executor_id, "failed", "ticket branch is missing")


def _ticket_has_priority(*, step: ExecutorStep, ctx: ExecutorContext, actor_result: dict[str, Any] | None) -> ExecutorStepResult:
    priority = ctx.ticket.get("priority")
    if _nonempty_str(priority):
        return ExecutorStepResult(step.executor_id, "passed", "ticket priority is present", {"priority": priority})
    return ExecutorStepResult(step.executor_id, "failed", "ticket priority is missing")


def _workspace_exists(*, step: ExecutorStep, ctx: ExecutorContext, actor_result: dict[str, Any] | None) -> ExecutorStepResult:
    path = Path(ctx.workspace_root)
    if path.exists() and path.is_dir():
        return ExecutorStepResult(step.executor_id, "passed", "workspace directory exists", {"path": str(path)})
    return ExecutorStepResult(step.executor_id, "error", "workspace directory does not exist", {"path": str(path)})


def _actor_completed(*, step: ExecutorStep, ctx: ExecutorContext, actor_result: dict[str, Any] | None) -> ExecutorStepResult:
    status = actor_result.get("status") if isinstance(actor_result, dict) else None
    if status == "completed":
        return ExecutorStepResult(step.executor_id, "passed", "actor completed", {"status": status})
    return ExecutorStepResult(step.executor_id, "failed", "actor did not complete", {"status": status})


def _evidence_has_any(*, step: ExecutorStep, ctx: ExecutorContext, actor_result: dict[str, Any] | None) -> ExecutorStepResult:
    raw_paths = step.params.get("paths")
    if not isinstance(raw_paths, list) or not all(isinstance(path, str) and path.strip() for path in raw_paths):
        return ExecutorStepResult(step.executor_id, "error", "evidence.has_any requires params.paths: list[str]")
    paths = [path.strip() for path in raw_paths]
    roots = _successful_prior_evidence_roots(ctx)
    if isinstance(actor_result, dict):
        roots.append(("actor_result", actor_result))
    roots.append(("ticket", ctx.ticket))
    for path in paths:
        for root_name, root in roots:
            found, value = _resolve_dotted_path(root, path)
            if found and value not in (None, "", [], {}):
                return ExecutorStepResult(
                    step.executor_id,
                    "passed",
                    f"{step.params.get('label') or 'evidence'} is present",
                    {"matched_path": path, "root": root_name},
                )
    return ExecutorStepResult(
        step.executor_id,
        "failed",
        f"{step.params.get('label') or 'evidence'} is missing",
        {"paths": paths},
    )


def _channel_require(*, step: ExecutorStep, ctx: ExecutorContext, actor_result: dict[str, Any] | None) -> ExecutorStepResult:
    channel = _nonempty_param(step.params, "channel")
    if not channel:
        return ExecutorStepResult(step.executor_id, "error", "channel.require requires params.channel")
    found, _value, root_name = _resolve_channel_value(channel=channel, ctx=ctx, actor_result=actor_result)
    if found:
        return ExecutorStepResult(
            step.executor_id,
            "passed",
            f"channel {channel} is present",
            {"channel": channel, "root": root_name},
        )
    return ExecutorStepResult(step.executor_id, "failed", f"channel {channel} is missing", {"channel": channel})


def _resolve_channel_value(
    *,
    channel: str,
    ctx: ExecutorContext,
    actor_result: dict[str, Any] | None,
) -> tuple[bool, Any, str | None]:
    roots = _successful_prior_evidence_roots(ctx)
    if isinstance(actor_result, dict):
        roots.append(("actor_result", actor_result))
    roots.append(("ticket", ctx.ticket))
    for root_name, root in roots:
        found, value = _resolve_dotted_path(root, channel)
        if not found and isinstance(root, dict) and isinstance(root.get("evidence"), dict):
            found, value = _resolve_dotted_path(root["evidence"], channel)
        if found and value not in (None, "", [], {}):
            return True, value, root_name
    return False, None, None


def _integration_dispatch(*, step: ExecutorStep, ctx: ExecutorContext, actor_result: dict[str, Any] | None) -> ExecutorStepResult:
    if not ctx.lease_id or ctx.workflow_client is None or not hasattr(ctx.workflow_client, "dispatch_integration"):
        return ExecutorStepResult(step.executor_id, "error", "integration.dispatch requires an active workflow runtime lease")
    provider = _nonempty_param(step.params, "provider")
    action = _nonempty_param(step.params, "action")
    if not provider or not action:
        return ExecutorStepResult(step.executor_id, "failed", "integration.dispatch requires params.provider and params.action")
    payload = step.params.get("payload") if isinstance(step.params.get("payload"), dict) else {}
    waterfall_pr = _extract_pr_reference_before_ticket(ctx=ctx, actor_result=actor_result)
    pr_payload = (
        {"pr_url": waterfall_pr["url"], "pr_links": [waterfall_pr["url"]]}
        if waterfall_pr is not None
        else _dispatch_pr_payload(ctx.ticket)
    )
    dispatch_payload = {
        "ticket_id": ctx.ticket.get("id"),
        "ticket_title": ctx.ticket.get("title"),
        **pr_payload,
        **payload,
    }
    try:
        result = ctx.workflow_client.dispatch_integration(
            ctx.lease_id,
            agent_id=ctx.agent_id,
            provider=provider,
            action=action,
            payload=dispatch_payload,
        )
    except Exception as err:  # noqa: BLE001 - external integration fault is executor evidence
        return ExecutorStepResult(step.executor_id, "error", f"integration dispatch failed: {err!r}")

    status = result.get("status")
    if status not in {"passed", "failed", "error", "skipped"}:
        return ExecutorStepResult(step.executor_id, "error", "integration dispatch returned invalid status")
    evidence = result.get("evidence") if isinstance(result.get("evidence"), dict) else {}
    return ExecutorStepResult(
        step.executor_id,
        status,  # type: ignore[arg-type]
        str(result.get("message") or "")[:4096],
        evidence,
    )


def _dispatch_pr_payload(ticket: dict[str, Any]) -> dict[str, Any]:
    pr_links = ticket.get("pr_links")
    if not isinstance(pr_links, list):
        return {}
    valid_links = [
        link.strip()
        for link in pr_links
        if isinstance(link, str) and _is_absolute_http_url(link.strip())
    ]
    if not valid_links:
        return {}
    return {
        "pr_url": valid_links[-1],
        "pr_links": valid_links,
    }


def _is_absolute_http_url(value: str) -> bool:
    if not value:
        return False
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _ci_run_command(
    *,
    step: ExecutorStep,
    ctx: ExecutorContext,
    command_kind: str,
    default_timeout: int,
    step_name: str | None = None,
) -> ExecutorStepResult:
    commands = _ci_commands(step.params)
    if not commands:
        return ExecutorStepResult(step.executor_id, "failed", f"{step.executor_id} requires params.command or params.commands")
    label = _ci_command_label(command_kind, step_name)
    workdir = _ci_working_directory(ctx, step.params.get("working_directory"))
    if workdir is None:
        return ExecutorStepResult(step.executor_id, "error", f"{step.executor_id} working_directory must stay inside the workspace")
    if not workdir.exists() or not workdir.is_dir():
        return ExecutorStepResult(step.executor_id, "error", f"{step.executor_id} working directory does not exist", {"working_directory": str(workdir)})

    timeout = step.timeout_seconds or default_timeout
    command_results: list[dict[str, Any]] = []
    for index, command in enumerate(commands):
        started = time.monotonic()
        try:
            proc = run_process(
                command,
                cwd=workdir,
                env=_controlled_env(ctx.env),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                check=False,
                shell=True,
            )
        except subprocess.TimeoutExpired as err:
            evidence = {
                "command": command,
                "command_index": index,
                "working_directory": str(workdir),
                "timeout_seconds": timeout,
                "command_kind": command_kind,
                **({"step_name": step_name} if step_name else {}),
                "stdout": _truncate(err.stdout if isinstance(err.stdout, str) else "", 64 * 1024),
                "stderr": _truncate(err.stderr if isinstance(err.stderr, str) else "", 64 * 1024),
                "commands": command_results,
            }
            return ExecutorStepResult(step.executor_id, "error", f"CI {label} command timed out after {timeout}s", evidence)

        elapsed_ms = int((time.monotonic() - started) * 1000)
        evidence = {
            "command": command,
            "command_index": index,
            "working_directory": str(workdir),
            "exit_code": int(proc.returncode),
            "duration_ms": elapsed_ms,
            "command_kind": command_kind,
            **({"step_name": step_name} if step_name else {}),
            "stdout": _truncate(proc.stdout or "", 64 * 1024),
            "stderr": _truncate(proc.stderr or "", 64 * 1024),
        }
        command_results.append(evidence)
        if proc.returncode != 0:
            return ExecutorStepResult(
                step.executor_id,
                "failed",
                f"CI {label} command failed with exit code {proc.returncode}",
                {**evidence, "commands": command_results},
            )

    final = command_results[-1]
    return ExecutorStepResult(
        step.executor_id,
        "passed",
        f"CI {label} command{'s' if len(commands) != 1 else ''} passed",
        {**final, "commands": command_results},
    )


def _ci_command_label(command_kind: str, step_name: str | None) -> str:
    if step_name:
        return step_name[:80]
    return command_kind


def _ci_custom_step_name(params: Mapping[str, Any]) -> str:
    for key in ("step_name", "label", "name"):
        value = params.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:80]
    return "Run command"


def _ci_run_tests(*, step: ExecutorStep, ctx: ExecutorContext, actor_result: dict[str, Any] | None) -> ExecutorStepResult:
    return _ci_run_command(step=step, ctx=ctx, command_kind="test", default_timeout=900)


def _ci_run_build(*, step: ExecutorStep, ctx: ExecutorContext, actor_result: dict[str, Any] | None) -> ExecutorStepResult:
    return _ci_run_command(step=step, ctx=ctx, command_kind="build", default_timeout=1200)


def _ci_run_typecheck(*, step: ExecutorStep, ctx: ExecutorContext, actor_result: dict[str, Any] | None) -> ExecutorStepResult:
    return _ci_run_command(step=step, ctx=ctx, command_kind="typecheck", default_timeout=900)


def _ci_run_custom_command(*, step: ExecutorStep, ctx: ExecutorContext, actor_result: dict[str, Any] | None) -> ExecutorStepResult:
    return _ci_run_command(
        step=step,
        ctx=ctx,
        command_kind="custom",
        default_timeout=900,
        step_name=_ci_custom_step_name(step.params),
    )


def _git_network_environment(
    *,
    step: ExecutorStep,
    ctx: ExecutorContext,
    access: GitHubAccess,
    evidence: dict[str, Any],
) -> dict[str, str] | ExecutorStepResult:
    expected_repo = _ticket_repo_slug(ctx.ticket)
    if expected_repo is None:
        return ExecutorStepResult(
            step.executor_id,
            "error",
            f"{step.executor_id} requires a GitHub repository slug for authenticated network access",
            evidence,
        )
    try:
        resolved = resolve_github_credential_environment(
            client=ctx.workflow_client,
            lease_id=ctx.lease_id,
            agent_id=ctx.agent_id,
            expected_repo=expected_repo,
            access=access,
            base_env=_controlled_env(ctx.env),
        )
    except GitHubCredentialEnvironmentError as err:
        return ExecutorStepResult(
            step.executor_id,
            "error",
            str(err),
            {**evidence, "credential_provider": "github_app"},
        )
    evidence["credential_provider"] = resolved.provider
    return resolved.env


def _git_ensure_ticket_branch(*, step: ExecutorStep, ctx: ExecutorContext, actor_result: dict[str, Any] | None) -> ExecutorStepResult:
    base_branch, target_branch = _ticket_branch_pair(step=step, ctx=ctx)
    if _uses_authoritative_validation_revisions(ctx):
        return _authoritative_validation_preflight_result(step=step, ctx=ctx)
    if not target_branch:
        return ExecutorStepResult(step.executor_id, "failed", "git.ensure_ticket_branch requires a ticket branch", {"base_branch": base_branch})
    if target_branch == base_branch:
        return ExecutorStepResult(
            step.executor_id,
            "failed",
            "git.ensure_ticket_branch target branch must differ from base branch",
            {"base_branch": base_branch, "target_branch": target_branch},
        )

    workdir = _ci_working_directory(ctx, step.params.get("working_directory"))
    if workdir is None:
        return ExecutorStepResult(step.executor_id, "error", "git.ensure_ticket_branch working_directory must stay inside the workspace")
    if not workdir.exists() or not workdir.is_dir():
        return ExecutorStepResult(step.executor_id, "error", "git.ensure_ticket_branch working directory does not exist", {"working_directory": str(workdir)})

    timeout = step.timeout_seconds or 300
    evidence: dict[str, Any] = {
        "base_branch": base_branch,
        "target_branch": target_branch,
        "branch": target_branch,
        "working_directory": str(workdir),
    }

    inside = _git_step(["rev-parse", "--is-inside-work-tree"], cwd=workdir, env=ctx.env, timeout=timeout)
    if inside.returncode != 0 or inside.stdout.strip() != "true":
        return ExecutorStepResult(step.executor_id, "error", "git.ensure_ticket_branch working directory is not a git checkout", {**evidence, **inside.evidence("git_probe")})

    before_branch = _git_step(["branch", "--show-current"], cwd=workdir, env=ctx.env, timeout=timeout)
    if before_branch.returncode == 0:
        evidence["before_branch"] = before_branch.stdout.strip()
    before_head = _git_step(["rev-parse", "HEAD"], cwd=workdir, env=ctx.env, timeout=timeout)
    if before_head.returncode == 0:
        evidence["before_head_sha"] = before_head.stdout.strip()

    dirty = _git_step(["status", "--porcelain"], cwd=workdir, env=ctx.env, timeout=timeout)
    if dirty.returncode != 0:
        return ExecutorStepResult(step.executor_id, "error", "git.ensure_ticket_branch could not inspect worktree status", {**evidence, **dirty.evidence("status")})
    if dirty.stdout.strip():
        # Contract B: dirt is classified, not lumped. Untracked paths matching
        # the state's declared generated-artifact allowlist are swept — they
        # are build output a preceding ci.run_* step produced and the repo
        # does not ignore. Everything else (tracked modifications, unknown
        # untracked paths) still fails: unknown dirt is work until proven
        # otherwise, and sweeping it would destroy evidence.
        swept, remaining = _sweep_generated_artifacts(
            step=step, ctx=ctx, workdir=workdir, status_stdout=dirty.stdout, timeout=timeout
        )
        if swept:
            evidence["swept_untracked"] = swept
        if remaining.strip():
            return ExecutorStepResult(step.executor_id, "failed", "git.ensure_ticket_branch requires a clean worktree before checkout: " + _dirty_worktree_summary(remaining), {**evidence, "dirty_status": _truncate(remaining, 8192)})

    network_env = _git_network_environment(step=step, ctx=ctx, access="read", evidence=evidence)
    if isinstance(network_env, ExecutorStepResult):
        return network_env

    fetch_base = _git_step(["fetch", "origin", f"{base_branch}:refs/remotes/origin/{base_branch}"], cwd=workdir, env=network_env, timeout=timeout)
    if fetch_base.returncode != 0:
        status: StepStatus = "error" if _is_github_auth_error(fetch_base.stderr) else "failed"
        return ExecutorStepResult(step.executor_id, status, f"git.ensure_ticket_branch failed to fetch {base_branch}", {**evidence, "failure_stage": "fetch_base", **fetch_base.evidence("fetch_base")})
    base_head = _git_step(["rev-parse", f"origin/{base_branch}"], cwd=workdir, env=ctx.env, timeout=timeout)
    if base_head.returncode != 0:
        return ExecutorStepResult(step.executor_id, "failed", "git.ensure_ticket_branch could not resolve base branch", {**evidence, **base_head.evidence("base_rev_parse")})
    evidence["base_sha"] = base_head.stdout.strip()

    fetch_target = _git_step(["fetch", "origin", f"{target_branch}:refs/remotes/origin/{target_branch}"], cwd=workdir, env=network_env, timeout=timeout)
    remote_exists = fetch_target.returncode == 0
    evidence["remote_existed"] = remote_exists
    if remote_exists:
        remote_head = _git_step(["rev-parse", f"origin/{target_branch}"], cwd=workdir, env=ctx.env, timeout=timeout)
        if remote_head.returncode != 0:
            return ExecutorStepResult(step.executor_id, "failed", "git.ensure_ticket_branch could not resolve remote ticket branch", {**evidence, **remote_head.evidence("target_rev_parse")})
        evidence["remote_head_sha"] = remote_head.stdout.strip()
    elif _git_remote_ref_missing(fetch_target):
        evidence.update(fetch_target.evidence("fetch_target"))
    else:
        return ExecutorStepResult(
            step.executor_id,
            "error",
            "git.ensure_ticket_branch could not determine whether the remote ticket branch exists",
            {**evidence, "failure_stage": "fetch_target", **fetch_target.evidence("fetch_target")},
        )

    local_branch = _git_step(["rev-parse", "--verify", f"refs/heads/{target_branch}"], cwd=workdir, env=ctx.env, timeout=timeout)
    local_exists = local_branch.returncode == 0
    evidence["local_existed"] = local_exists
    created = not local_exists and not remote_exists

    if local_exists:
        checkout = _git_step(["checkout", target_branch], cwd=workdir, env=ctx.env, timeout=timeout)
    elif remote_exists:
        checkout = _git_step(["checkout", "-B", target_branch, f"origin/{target_branch}"], cwd=workdir, env=ctx.env, timeout=timeout)
    else:
        checkout = _git_step(["checkout", "-B", target_branch, f"origin/{base_branch}"], cwd=workdir, env=ctx.env, timeout=timeout)
    if checkout.returncode != 0:
        return ExecutorStepResult(step.executor_id, "failed", "git.ensure_ticket_branch could not checkout ticket branch", {**evidence, "failure_stage": "checkout", **checkout.evidence("checkout")})

    if local_exists and remote_exists:
        current_head = _git_step(["rev-parse", "HEAD"], cwd=workdir, env=ctx.env, timeout=timeout)
        if current_head.returncode != 0:
            return ExecutorStepResult(step.executor_id, "error", "git.ensure_ticket_branch could not resolve checked-out branch", {**evidence, **current_head.evidence("head_rev_parse")})
        local_sha = current_head.stdout.strip()
        remote_sha = str(evidence.get("remote_head_sha") or "")
        if local_sha != remote_sha:
            remote_contains_local = _git_step(["merge-base", "--is-ancestor", "HEAD", f"origin/{target_branch}"], cwd=workdir, env=ctx.env, timeout=timeout)
            if remote_contains_local.returncode == 0:
                ff = _git_step(["merge", "--ff-only", f"origin/{target_branch}"], cwd=workdir, env=ctx.env, timeout=timeout)
                if ff.returncode != 0:
                    return ExecutorStepResult(step.executor_id, "failed", "git.ensure_ticket_branch could not fast-forward local ticket branch", {**evidence, "failure_stage": "fast_forward", **ff.evidence("fast_forward")})
                evidence["fast_forwarded"] = True
            else:
                local_contains_remote = _git_step(["merge-base", "--is-ancestor", f"origin/{target_branch}", "HEAD"], cwd=workdir, env=ctx.env, timeout=timeout)
                if local_contains_remote.returncode != 0:
                    return ExecutorStepResult(
                        step.executor_id,
                        "failed",
                        "git.ensure_ticket_branch local ticket branch has diverged from origin",
                        {**evidence, "failure_stage": "divergence_check", **local_contains_remote.evidence("ancestor_check")},
                    )
                evidence["fast_forwarded"] = False
    else:
        evidence["fast_forwarded"] = False

    after_branch = _git_step(["branch", "--show-current"], cwd=workdir, env=ctx.env, timeout=timeout)
    if after_branch.returncode != 0:
        return ExecutorStepResult(step.executor_id, "error", "git.ensure_ticket_branch could not resolve current branch after checkout", {**evidence, **after_branch.evidence("after_branch")})
    checked_out = after_branch.stdout.strip()
    if checked_out != target_branch:
        return ExecutorStepResult(step.executor_id, "error", "git.ensure_ticket_branch did not end on the ticket branch", {**evidence, "current_branch": checked_out})

    after_head = _git_step(["rev-parse", "HEAD"], cwd=workdir, env=ctx.env, timeout=timeout)
    if after_head.returncode != 0:
        return ExecutorStepResult(step.executor_id, "error", "git.ensure_ticket_branch could not resolve final HEAD", {**evidence, **after_head.evidence("after_rev_parse")})

    return ExecutorStepResult(
        step.executor_id,
        "passed",
        "ticket branch checkout is ready",
        {**evidence, "current_branch": checked_out, "after_head_sha": after_head.stdout.strip(), "created": created},
    )


def _git_sync_with_base(*, step: ExecutorStep, ctx: ExecutorContext, actor_result: dict[str, Any] | None) -> ExecutorStepResult:
    mode = str(step.params.get("mode") or "merge").strip().lower()
    if mode not in {"merge", "rebase"}:
        return ExecutorStepResult(step.executor_id, "error", "git.sync_with_base mode must be merge or rebase", {"mode": mode})
    if _uses_authoritative_validation_revisions(ctx):
        return _authoritative_validation_preflight_result(step=step, ctx=ctx)
    base_branch = _branch_param(step.params.get("base_branch")) or _branch_param(ctx.ticket.get("base_branch")) or "main"
    target_branch = (
        _branch_param(step.params.get("target_branch"))
        or _branch_param(ctx.ticket.get("branch"))
        or _branch_param(ctx.ticket.get("ticket_branch"))
        or _branch_param(ctx.ticket.get("head_branch"))
    )
    if not target_branch:
        return ExecutorStepResult(step.executor_id, "failed", "git.sync_with_base requires a ticket branch", {"base_branch": base_branch})
    if target_branch == base_branch:
        return ExecutorStepResult(
            step.executor_id,
            "failed",
            "git.sync_with_base target branch must differ from base branch",
            {"base_branch": base_branch, "target_branch": target_branch},
        )
    workdir = _ci_working_directory(ctx, step.params.get("working_directory"))
    if workdir is None:
        return ExecutorStepResult(step.executor_id, "error", "git.sync_with_base working_directory must stay inside the workspace")
    if not workdir.exists() or not workdir.is_dir():
        return ExecutorStepResult(step.executor_id, "error", "git.sync_with_base working directory does not exist", {"working_directory": str(workdir)})

    timeout = step.timeout_seconds or 300
    evidence: dict[str, Any] = {
        "mode": mode,
        "base_branch": base_branch,
        "target_branch": target_branch,
        "working_directory": str(workdir),
    }
    delegate_conflicts = str(ctx.ticket.get("governance_state_type") or "").strip().lower() == "implementation"
    evidence["conflict_policy"] = "delegate" if delegate_conflicts else "fail"

    inside = _git_step(["rev-parse", "--is-inside-work-tree"], cwd=workdir, env=ctx.env, timeout=timeout)
    if inside.returncode != 0 or inside.stdout.strip() != "true":
        return ExecutorStepResult(step.executor_id, "error", "git.sync_with_base working directory is not a git checkout", {**evidence, **inside.evidence("git_probe")})

    dirty = _git_step(["status", "--porcelain"], cwd=workdir, env=ctx.env, timeout=timeout)
    if dirty.returncode != 0:
        return ExecutorStepResult(step.executor_id, "error", "git.sync_with_base could not inspect worktree status", {**evidence, **dirty.evidence("status")})
    if dirty.stdout.strip():
        network_env = _git_network_environment(step=step, ctx=ctx, access="write", evidence=evidence)
        if isinstance(network_env, ExecutorStepResult):
            return network_env
        fetched_base = _git_step(["fetch", "origin", f"{base_branch}:refs/remotes/origin/{base_branch}"], cwd=workdir, env=network_env, timeout=timeout)
        if fetched_base.returncode != 0:
            return ExecutorStepResult(step.executor_id, "failed", "git.sync_with_base could not refresh the configured base before delegation", {**evidence, **fetched_base.evidence("fetch_base")})
        base_head = _git_step(["rev-parse", f"origin/{base_branch}"], cwd=workdir, env=ctx.env, timeout=timeout)
        if base_head.returncode != 0:
            return ExecutorStepResult(step.executor_id, "failed", "git.sync_with_base could not resolve the configured base before delegation", {**evidence, **base_head.evidence("base_rev_parse")})
        evidence["required_base_sha"] = base_head.stdout.strip()
        merge_head = _git_step(["rev-parse", "--verify", "MERGE_HEAD"], cwd=workdir, env=ctx.env, timeout=timeout)
        rebase_head = _git_step(["rev-parse", "--verify", "REBASE_HEAD"], cwd=workdir, env=ctx.env, timeout=timeout)
        reconciliation_in_progress = merge_head.returncode == 0 or rebase_head.returncode == 0
        if delegate_conflicts and reconciliation_in_progress:
            conflicts = _git_step(["diff", "--name-only", "--diff-filter=U"], cwd=workdir, env=ctx.env, timeout=timeout)
            conflict_paths = [line.strip() for line in conflicts.stdout.splitlines() if line.strip()]
            return ExecutorStepResult(
                step.executor_id,
                "passed",
                "base synchronization conflict delegated to implementation agent",
                {
                    **evidence,
                    "base_sync_required": True,
                    "reconciliation_in_progress": True,
                    "conflict_paths": conflict_paths,
                    "failure_stage": "rebase" if rebase_head.returncode == 0 else "merge",
                },
            )
        if delegate_conflicts:
            return ExecutorStepResult(
                step.executor_id,
                "passed",
                "base synchronization delegated because the implementation workspace contains retained work",
                {
                    **evidence,
                    "base_sync_required": True,
                    "reconciliation_in_progress": False,
                    "dirty_status": _truncate(dirty.stdout, 8192),
                    "delegation_reason": "retained_workspace_work",
                },
            )
        return ExecutorStepResult(step.executor_id, "failed", "git.sync_with_base requires a clean worktree: " + _dirty_worktree_summary(dirty.stdout), {**evidence, "dirty_status": _truncate(dirty.stdout, 8192)})

    network_env = _git_network_environment(step=step, ctx=ctx, access="write", evidence=evidence)
    if isinstance(network_env, ExecutorStepResult):
        return network_env

    for branch, stage in ((target_branch, "fetch_target"), (base_branch, "fetch_base")):
        fetched = _git_step(["fetch", "origin", f"{branch}:refs/remotes/origin/{branch}"], cwd=workdir, env=network_env, timeout=timeout)
        if (
            fetched.returncode != 0
            and evidence.get("credential_provider") == "github_app"
            and not evidence.get("credential_refresh_attempted")
            and _is_github_auth_error(fetched.stderr)
        ):
            # Installation tokens are deliberately ephemeral.  If GitHub rejects
            # one, discard it and ask the service for a newly minted credential
            # before deciding that the workflow cannot continue.  Never persist
            # the token, its encoding, or a fingerprint in executor evidence.
            evidence["credential_refresh_attempted"] = True
            evidence["network_failure_category"] = "github_authentication"
            refreshed_env = _git_network_environment(step=step, ctx=ctx, access="write", evidence=evidence)
            if isinstance(refreshed_env, ExecutorStepResult):
                return refreshed_env
            network_env = refreshed_env
            fetched = _git_step(["fetch", "origin", f"{branch}:refs/remotes/origin/{branch}"], cwd=workdir, env=network_env, timeout=timeout)
            evidence["credential_refresh_succeeded"] = fetched.returncode == 0
        if fetched.returncode != 0:
            status: StepStatus = "error" if _is_github_auth_error(fetched.stderr) else "failed"
            return ExecutorStepResult(step.executor_id, status, f"git.sync_with_base failed to fetch {branch}", {**evidence, "failure_stage": stage, **fetched.evidence(stage)})

    before = _git_step(["rev-parse", f"origin/{target_branch}"], cwd=workdir, env=ctx.env, timeout=timeout)
    base = _git_step(["rev-parse", f"origin/{base_branch}"], cwd=workdir, env=ctx.env, timeout=timeout)
    if before.returncode != 0 or base.returncode != 0:
        return ExecutorStepResult(step.executor_id, "failed", "git.sync_with_base could not resolve branch SHAs", {**evidence, **before.evidence("target_rev_parse"), **base.evidence("base_rev_parse")})
    before_sha = before.stdout.strip()
    base_sha = base.stdout.strip()
    evidence.update({"before_head_sha": before_sha, "base_sha": base_sha})

    contains_base = _git_step(["merge-base", "--is-ancestor", f"origin/{base_branch}", f"origin/{target_branch}"], cwd=workdir, env=ctx.env, timeout=timeout)
    if contains_base.returncode == 0:
        return ExecutorStepResult(step.executor_id, "passed", "ticket branch already contains base branch", {**evidence, "after_head_sha": before_sha, "updated": False})

    # `checkout -B <branch> origin/<branch>` force-moves the local branch to the
    # remote tip, which silently destroys commits that have not been pushed yet.
    # The agent commits its work and only the publish step pushes, so any
    # failure between the two (a dirty worktree, a failed check) leaves the work
    # local-only — and the next attempt's sync then threw it away. Check out the
    # existing branch instead and let the merge/rebase below carry base into it.
    local_ref = _git_step(["rev-parse", "--verify", f"refs/heads/{target_branch}"], cwd=workdir, env=ctx.env, timeout=timeout)
    unpushed = 0
    if local_ref.returncode == 0:
        counted = _git_step(["rev-list", "--count", f"origin/{target_branch}..{target_branch}"], cwd=workdir, env=ctx.env, timeout=timeout)
        if counted.returncode == 0:
            try:
                unpushed = int(counted.stdout.strip() or "0")
            except ValueError:
                unpushed = 0
    evidence["unpushed_local_commits"] = unpushed
    if unpushed > 0:
        checkout = _git_step(["checkout", target_branch], cwd=workdir, env=ctx.env, timeout=timeout)
    else:
        checkout = _git_step(["checkout", "-B", target_branch, f"origin/{target_branch}"], cwd=workdir, env=ctx.env, timeout=timeout)
    if checkout.returncode != 0:
        return ExecutorStepResult(step.executor_id, "failed", "git.sync_with_base could not checkout target branch", {**evidence, "failure_stage": "checkout", **checkout.evidence("checkout")})

    if mode == "rebase":
        sync = _git_step(["rebase", f"origin/{base_branch}"], cwd=workdir, env=ctx.env, timeout=timeout)
        failure_stage = "rebase"
    else:
        sync = _git_step([
            "-c", "user.name=SICKR Workflow",
            "-c", "user.email=workflow@sickr.ai",
            "merge", "--no-edit", f"origin/{base_branch}",
        ], cwd=workdir, env=ctx.env, timeout=timeout)
        failure_stage = "merge"
    if sync.returncode != 0:
        conflicts = _git_step(["diff", "--name-only", "--diff-filter=U"], cwd=workdir, env=ctx.env, timeout=timeout)
        conflict_paths = [line.strip() for line in conflicts.stdout.splitlines() if line.strip()]
        if delegate_conflicts and conflict_paths:
            return ExecutorStepResult(
                step.executor_id,
                "passed",
                "base synchronization conflict delegated to implementation agent",
                {
                    **evidence,
                    "base_sync_required": True,
                    "reconciliation_in_progress": True,
                    "conflict_paths": conflict_paths,
                    "failure_stage": failure_stage,
                    **sync.evidence(failure_stage),
                },
            )
        abort_cmd = ["rebase", "--abort"] if mode == "rebase" else ["merge", "--abort"]
        aborted = _git_step(abort_cmd, cwd=workdir, env=ctx.env, timeout=timeout)
        return ExecutorStepResult(
            step.executor_id,
            "failed",
            f"git.sync_with_base {mode} failed; agent assistance may be required",
            {**evidence, "failure_stage": failure_stage, **sync.evidence(failure_stage), **aborted.evidence(f"{failure_stage}_abort")},
        )

    after = _git_step(["rev-parse", "HEAD"], cwd=workdir, env=ctx.env, timeout=timeout)
    if after.returncode != 0:
        return ExecutorStepResult(step.executor_id, "error", "git.sync_with_base could not resolve updated HEAD", {**evidence, **after.evidence("after_rev_parse")})
    after_sha = after.stdout.strip()
    push = _git_step(["push", "origin", f"HEAD:{target_branch}"], cwd=workdir, env=network_env, timeout=timeout)
    if push.returncode != 0:
        return ExecutorStepResult(step.executor_id, "failed", "git.sync_with_base could not push updated ticket branch", {**evidence, "after_head_sha": after_sha, "failure_stage": "push", **push.evidence("push")})

    return ExecutorStepResult(
        step.executor_id,
        "passed",
        "ticket branch updated from base branch",
        {**evidence, "after_head_sha": after_sha, "updated": after_sha != before_sha, **push.evidence("push")},
    )


def _git_ensure_published(*, step: ExecutorStep, ctx: ExecutorContext, actor_result: dict[str, Any] | None) -> ExecutorStepResult:
    base_branch, target_branch = _ticket_branch_pair(step=step, ctx=ctx)
    if not target_branch:
        return ExecutorStepResult(step.executor_id, "failed", "git.ensure_published requires a ticket branch", {"base_branch": base_branch})
    if target_branch == base_branch:
        return ExecutorStepResult(
            step.executor_id,
            "failed",
            "git.ensure_published refuses to publish the base branch",
            {"base_branch": base_branch, "target_branch": target_branch},
        )
    workdir = _ci_working_directory(ctx, step.params.get("working_directory"))
    if workdir is None:
        return ExecutorStepResult(step.executor_id, "error", "git.ensure_published working_directory must stay inside the workspace")
    if not workdir.exists() or not workdir.is_dir():
        return ExecutorStepResult(step.executor_id, "error", "git.ensure_published working directory does not exist", {"working_directory": str(workdir)})

    timeout = step.timeout_seconds or 300
    evidence: dict[str, Any] = {
        "base_branch": base_branch,
        "target_branch": target_branch,
        "branch": target_branch,
        "working_directory": str(workdir),
    }

    inside = _git_step(["rev-parse", "--is-inside-work-tree"], cwd=workdir, env=ctx.env, timeout=timeout)
    if inside.returncode != 0 or inside.stdout.strip() != "true":
        return ExecutorStepResult(step.executor_id, "error", "git.ensure_published working directory is not a git checkout", {**evidence, **inside.evidence("git_probe")})

    current_branch = _git_step(["branch", "--show-current"], cwd=workdir, env=ctx.env, timeout=timeout)
    if current_branch.returncode != 0:
        return ExecutorStepResult(step.executor_id, "error", "git.ensure_published could not resolve current branch", {**evidence, **current_branch.evidence("current_branch")})
    checked_out = current_branch.stdout.strip()
    evidence["current_branch"] = checked_out
    if checked_out != target_branch:
        return ExecutorStepResult(step.executor_id, "failed", "git.ensure_published must run on the ticket branch checkout", evidence)

    dirty = _git_step(["status", "--porcelain"], cwd=workdir, env=ctx.env, timeout=timeout)
    if dirty.returncode != 0:
        return ExecutorStepResult(step.executor_id, "error", "git.ensure_published could not inspect worktree status", {**evidence, **dirty.evidence("status")})
    if dirty.stdout.strip():
        return ExecutorStepResult(step.executor_id, "failed", "git.ensure_published requires a clean worktree: " + _dirty_worktree_summary(dirty.stdout), {**evidence, "dirty_status": _truncate(dirty.stdout, 8192)})

    head = _git_step(["rev-parse", "HEAD"], cwd=workdir, env=ctx.env, timeout=timeout)
    if head.returncode != 0:
        return ExecutorStepResult(step.executor_id, "error", "git.ensure_published could not resolve HEAD", {**evidence, **head.evidence("head_rev_parse")})
    head_sha = head.stdout.strip()
    evidence["head_sha"] = head_sha

    network_env = _git_network_environment(step=step, ctx=ctx, access="write", evidence=evidence)
    if isinstance(network_env, ExecutorStepResult):
        return network_env

    fetch = _git_step(["fetch", "origin", f"{target_branch}:refs/remotes/origin/{target_branch}"], cwd=workdir, env=network_env, timeout=timeout)
    remote_exists = fetch.returncode == 0
    evidence["remote_existed"] = remote_exists
    if remote_exists:
        remote_head = _git_step(["rev-parse", f"origin/{target_branch}"], cwd=workdir, env=ctx.env, timeout=timeout)
        if remote_head.returncode != 0:
            return ExecutorStepResult(step.executor_id, "failed", "git.ensure_published could not resolve remote ticket branch", {**evidence, **remote_head.evidence("remote_rev_parse")})
        remote_sha = remote_head.stdout.strip()
        evidence["remote_head_sha"] = remote_sha
        if remote_sha == head_sha:
            return ExecutorStepResult(step.executor_id, "passed", "ticket branch already published", {**evidence, "pushed": False})
        ancestor = _git_step(["merge-base", "--is-ancestor", f"origin/{target_branch}", "HEAD"], cwd=workdir, env=ctx.env, timeout=timeout)
        if ancestor.returncode != 0:
            return ExecutorStepResult(
                step.executor_id,
                "failed",
                "git.ensure_published local branch is not a fast-forward of the remote ticket branch",
                {**evidence, **ancestor.evidence("ancestor_check")},
            )
    elif _git_remote_ref_missing(fetch):
        evidence.update(fetch.evidence("fetch_remote"))
    else:
        return ExecutorStepResult(
            step.executor_id,
            "error",
            "git.ensure_published could not determine whether the remote ticket branch exists",
            {**evidence, "failure_stage": "fetch_remote", **fetch.evidence("fetch_remote")},
        )

    # Contract B: pushing the agent's unpushed branch is a completion action.
    # The concrete checks above (clean tree, on-branch, fast-forward ancestry)
    # are the authorization inputs; the manifest names that set in one place
    # and fails closed if a future edit drops one.
    preconditions = {
        "clean_worktree": True,
        "on_ticket_branch": True,
        "remote_fast_forward_or_absent": True,
    }
    decision = authorize_completion(
        "git.push_unpushed_branch",
        evidence={"target_branch": target_branch, "head_sha": head_sha},
        preconditions=preconditions,
    )
    if not decision.allowed:
        return ExecutorStepResult(
            step.executor_id,
            "error",
            f"completion action denied: {decision.reason}",
            {**evidence, "completion_denied": decision.as_dict()},
        )
    push = _git_step(["push", "origin", f"HEAD:{target_branch}"], cwd=workdir, env=network_env, timeout=timeout)
    if push.returncode != 0:
        return ExecutorStepResult(step.executor_id, "failed", "git.ensure_published could not push ticket branch", {**evidence, **push.evidence("push")})

    audit = completion_audit_record(
        action_id="git.push_unpushed_branch",
        before={"remote_head_sha": evidence.get("remote_head_sha"), "remote_existed": remote_exists},
        after={"remote_head_sha": head_sha},
        evidence_inputs={"target_branch": target_branch, "head_sha": head_sha},
        preconditions_checked=preconditions,
        idempotency_key=f"{target_branch}@{head_sha}",
    )
    return ExecutorStepResult(
        step.executor_id,
        "passed",
        "ticket branch published",
        {**evidence, "remote_head_sha": head_sha, "pushed": True, "completion_audit": audit, **push.evidence("push")},
    )


def _package_install_dependencies(*, step: ExecutorStep, ctx: ExecutorContext, actor_result: dict[str, Any] | None) -> ExecutorStepResult:
    """Install from the repository's lockfile, unless commands are explicit."""
    if _ci_commands(step.params):
        return _ci_run_command(step=step, ctx=ctx, command_kind="dependency_install", default_timeout=1200)
    workdir = _ci_working_directory(ctx, step.params.get("working_directory"))
    if workdir is None or not workdir.is_dir():
        return ExecutorStepResult(step.executor_id, "error", "package.install_dependencies requires a repository workspace")
    command = None
    for marker, candidate in (
        ("pnpm-lock.yaml", "pnpm install --frozen-lockfile"),
        ("yarn.lock", "yarn install --frozen-lockfile"),
        ("package-lock.json", "npm ci"),
        ("uv.lock", "uv sync --frozen"),
        ("poetry.lock", "poetry install --no-interaction"),
        ("requirements.txt", f'"{sys.executable}" -m pip install -r requirements.txt'),
    ):
        if (workdir / marker).exists():
            command = candidate
            break
    if command is None:
        return ExecutorStepResult(step.executor_id, "passed", "repository has no dependency installation step", {"working_directory": str(workdir), "installed": False})
    inferred = replace(step, params={**step.params, "command": command})
    return _ci_run_command(step=inferred, ctx=ctx, command_kind="dependency_install", default_timeout=1200)


def _workspace_clone_repositories(*, step: ExecutorStep, ctx: ExecutorContext, actor_result: dict[str, Any] | None) -> ExecutorStepResult:
    """Materialize the primary ticket repository in its allocated workspace."""
    del actor_result
    destination_raw = str(ctx.env.get("TICKET_WORKSPACE") or ctx.workspace_root / "ticket-repo").strip()
    repo_slug = str(ctx.ticket.get("repo") or "").strip()
    remote_url = str(ctx.env.get("TICKET_REMOTE_URL") or ctx.ticket.get("repo_url") or (f"https://github.com/{repo_slug}.git" if repo_slug else "")).strip()
    if not destination_raw or not remote_url:
        return ExecutorStepResult(step.executor_id, "error", "workspace.clone_repositories requires allocated workspace and remote URL")
    destination = Path(destination_raw).resolve()
    root = ctx.workspace_root.resolve()
    try:
        destination.relative_to(root)
    except ValueError:
        return ExecutorStepResult(step.executor_id, "error", "workspace.clone_repositories destination escapes the runtime workspace")
    requested: list[tuple[str, str, Path, str]] = [(repo_slug or "primary", remote_url, destination, str(ctx.ticket.get("base_branch") or "main"))]
    for raw in ctx.ticket.get("context_repos") or []:
        if not isinstance(raw, dict):
            continue
        slug = str(raw.get("repo") or "").strip()
        url = str(raw.get("repo_url") or (f"https://github.com/{slug}.git" if slug else "")).strip()
        if not slug or not url or slug == repo_slug:
            continue
        target = (destination.parent / slug.rsplit("/", 1)[-1]).resolve()
        try:
            target.relative_to(destination.parent.resolve())
        except ValueError:
            return ExecutorStepResult(step.executor_id, "error", "workspace.clone_repositories context destination escapes the runtime workspace", {"repo": slug})
        if target == destination or any(target == item[2] for item in requested):
            return ExecutorStepResult(step.executor_id, "error", "workspace.clone_repositories repository destinations must be unique", {"repo": slug, "workspace_path": str(target)})
        requested.append((slug, url, target, str(raw.get("base_branch") or raw.get("default_branch") or "main")))
    records: list[dict[str, Any]] = []
    for slug, url, target, base_branch in requested:
        existed = (target / ".git").exists()
        if existed:
            actual_remote = _git_step(["remote", "get-url", "origin"], cwd=target, env=ctx.env, timeout=120)
            if actual_remote.returncode != 0 or (github_repo_from_remote(actual_remote.stdout.strip()) or "").casefold() != slug.casefold():
                return ExecutorStepResult(step.executor_id, "failed", "workspace.clone_repositories existing checkout has the wrong origin", {"repo": slug, "workspace_path": str(target)})
            retry_status = _git_step(["status", "--porcelain=v1", "-z", "--untracked-files=all"], cwd=target, env=ctx.env, timeout=120)
            if retry_status.returncode != 0:
                return ExecutorStepResult(step.executor_id, "error", f"workspace.clone_repositories could not inspect existing {slug} checkout", retry_status.evidence("retry_status"))
            if retry_status.stdout:
                records.append({"repo": slug, "workspace_path": str(target), "remote_url": url, "base_branch": base_branch, "cloned": False, "reused_dirty": True, "reused_unpublished": False})
                continue
            unpublished = False
            unpublished_basis = ""
            upstream = _git_step(["rev-list", "--count", "@{upstream}..HEAD"], cwd=target, env=ctx.env, timeout=120)
            upstream_known = upstream.returncode == 0
            if upstream_known:
                try:
                    unpublished = int(upstream.stdout.strip() or "0") > 0
                    unpublished_basis = "upstream"
                except ValueError:
                    return ExecutorStepResult(step.executor_id, "error", f"workspace.clone_repositories received an invalid unpublished commit count for {slug}")
            if not upstream_known:
                branch = _git_step(["branch", "--show-current"], cwd=target, env=ctx.env, timeout=120)
                refs = [f"origin/{branch.stdout.strip()}"] if branch.returncode == 0 and branch.stdout.strip() else []
                refs.append(f"origin/{base_branch}")
                for remote_ref in dict.fromkeys(refs):
                    relative = _git_step(["rev-list", "--count", f"{remote_ref}..HEAD"], cwd=target, env=ctx.env, timeout=120)
                    if relative.returncode != 0:
                        continue
                    try:
                        if int(relative.stdout.strip() or "0") > 0:
                            unpublished = True
                            unpublished_basis = remote_ref
                    except ValueError:
                        return ExecutorStepResult(step.executor_id, "error", f"workspace.clone_repositories received an invalid unpublished commit count for {slug}")
                    break
            if unpublished:
                records.append({
                    "repo": slug,
                    "workspace_path": str(target),
                    "remote_url": url,
                    "base_branch": base_branch,
                    "cloned": False,
                    "reused_dirty": False,
                    "reused_unpublished": unpublished,
                    **({"unpublished_basis": unpublished_basis} if unpublished_basis else {}),
                })
                continue
        if not existed and target.exists() and any(target.iterdir()):
            return ExecutorStepResult(step.executor_id, "failed", "workspace.clone_repositories refuses a non-empty non-Git destination", {"repo": slug, "workspace_path": str(target), "repositories": records})
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            credential = resolve_github_credential_environment(
                client=ctx.workflow_client,
                lease_id=ctx.lease_id,
                agent_id=ctx.agent_id,
                expected_repo=slug,
                access="read",
                base_env=_controlled_env(ctx.env),
            )
        except GitHubCredentialEnvironmentError as err:
            return ExecutorStepResult(step.executor_id, "error", str(err), {"repo": slug, "repositories": records})
        if not existed:
            cloned = _git_step(["clone", "--no-checkout", url, str(target)], cwd=target.parent, env=credential.env, timeout=_bounded_timeout(step.timeout_seconds) or 300)
            if cloned.returncode != 0:
                return ExecutorStepResult(step.executor_id, "failed", f"workspace.clone_repositories could not clone {slug}", {**cloned.evidence("git_clone"), "repo": slug, "repositories": records})
            # Real Git creates these; explicit creation keeps injected command
            # runners and filesystem-backed tests faithful to the next steps.
            (target / ".git").mkdir(parents=True, exist_ok=True)
        fetched = _git_step(["fetch", "origin", f"{base_branch}:refs/remotes/origin/{base_branch}"], cwd=target, env=credential.env, timeout=300)
        if fetched.returncode != 0:
            return ExecutorStepResult(step.executor_id, "failed", f"workspace.clone_repositories could not fetch {slug} base", {**fetched.evidence("git_fetch"), "repo": slug, "base_branch": base_branch})
        checkout = _git_step(["checkout", "--detach", f"origin/{base_branch}"], cwd=target, env=credential.env, timeout=120)
        if checkout.returncode != 0:
            return ExecutorStepResult(step.executor_id, "failed", f"workspace.clone_repositories could not checkout {slug} base", checkout.evidence("git_checkout"))
        cleaned = _git_step(["clean", "-fdx"], cwd=target, env=credential.env, timeout=120)
        if cleaned.returncode != 0:
            return ExecutorStepResult(step.executor_id, "failed", f"workspace.clone_repositories could not clean {slug}", cleaned.evidence("git_clean"))
        baseline = _git_step(["status", "--porcelain=v1", "-z", "--untracked-files=all"], cwd=target, env=credential.env, timeout=120)
        if baseline.returncode != 0 or baseline.stdout:
            return ExecutorStepResult(step.executor_id, "failed", f"workspace.clone_repositories did not produce a clean {slug} base", baseline.evidence("baseline_status"))
        (target / ".git" / "sickr-setup-baseline").write_bytes(baseline.stdout.encode())
        marker = target / ".git" / "sickr-workspace-generation.json"
        if not marker.exists():
            ledger = target.parent / ".sickr-workspace-generations" / f"{target.name}.json"
            generation = 0
            try:
                saved = json.loads(ledger.read_text(encoding="utf-8"))
                generation = int(saved.get("generation") or 0) if isinstance(saved, dict) else 0
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass
            generation += 1
            ledger.parent.mkdir(parents=True, exist_ok=True)
            ledger.write_text(json.dumps({"generation": generation}) + "\n", encoding="utf-8")
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(json.dumps({"generation": generation}) + "\n", encoding="utf-8")
        records.append({"repo": slug, "workspace_path": str(target), "remote_url": url, "base_branch": base_branch, "cloned": not existed})
    (destination.parent / ".sickr-primary-workspace").write_text(destination.name + "\n", encoding="utf-8")
    return ExecutorStepResult(step.executor_id, "passed", "required repositories materialized", {"workspace_path": str(destination), "remote_url": remote_url, "cloned": any(r["cloned"] for r in records), "repositories": records})


def _validation_run_baseline(*, step: ExecutorStep, ctx: ExecutorContext, actor_result: dict[str, Any] | None) -> ExecutorStepResult:
    """Run an explicit baseline command or infer the package's standard test."""
    workdir = _ci_working_directory(ctx, step.params.get("working_directory"))
    if workdir is None or not workdir.is_dir():
        return ExecutorStepResult(step.executor_id, "error", "validation.run_baseline requires a repository workspace")
    passed_marker = workdir / ".git" / "sickr-baseline-passed.json"
    passed_marker.unlink(missing_ok=True)
    effective = step
    if not _ci_commands(step.params):
        if (workdir / "package.json").exists():
            command = "npm test"
        elif (workdir / "pyproject.toml").exists() or (workdir / "pytest.ini").exists():
            command = f'"{sys.executable}" -m pytest -q'
        else:
            return ExecutorStepResult(step.executor_id, "failed", "repository has no configured or inferable baseline test command", {"working_directory": str(workdir)})
        effective = replace(step, params={**step.params, "command": command})
    result = _ci_run_command(step=effective, ctx=ctx, command_kind="baseline", default_timeout=1200)
    if result.status == "passed":
        head = _git_step(["rev-parse", "HEAD"], cwd=workdir, env=ctx.env, timeout=120)
        if head.returncode != 0:
            return ExecutorStepResult(step.executor_id, "error", "validation.run_baseline could not identify the validated revision", head.evidence("git_head"))
        passed_marker.write_text(json.dumps({"head_sha": head.stdout.strip()}) + "\n", encoding="utf-8")
    return result


def _workspace_capture_generated_ignores(*, step: ExecutorStep, ctx: ExecutorContext, actor_result: dict[str, Any] | None) -> ExecutorStepResult:
    """Locally exclude untracked paths produced by successful Setup flights."""
    workdir = _ci_working_directory(ctx, step.params.get("working_directory"))
    if workdir is None or not workdir.is_dir():
        return ExecutorStepResult(step.executor_id, "error", "workspace.capture_generated_ignores requires a repository workspace")
    passed_marker = workdir / ".git" / "sickr-baseline-passed.json"
    try:
        passed_head = str(json.loads(passed_marker.read_text(encoding="utf-8")).get("head_sha") or "")
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        passed_head = ""
    current_head = _git_step(["rev-parse", "HEAD"], cwd=workdir, env=ctx.env, timeout=120)
    if current_head.returncode != 0 or not passed_head or passed_head != current_head.stdout.strip():
        return ExecutorStepResult(step.executor_id, "failed", "workspace.capture_generated_ignores requires a successful baseline for the current revision")
    status = _git_step(["status", "--porcelain=v1", "-z", "--untracked-files=all"], cwd=workdir, env=ctx.env, timeout=120)
    if status.returncode != 0:
        return ExecutorStepResult(step.executor_id, "error", "could not inspect setup-generated files", status.evidence("status"))
    entries = [entry for entry in status.stdout.split("\0") if entry]
    tracked = [entry for entry in entries if not entry.startswith("?? ")]
    if tracked:
        return ExecutorStepResult(step.executor_id, "failed", "Setup install/test modified tracked files; generated exclusions refused", {"tracked_changes": tracked[:100]})
    baseline_file = workdir / ".git" / "sickr-setup-baseline"
    baseline_entries = set(baseline_file.read_bytes().decode(errors="replace").split("\0")) if baseline_file.exists() else set()
    paths = [entry[3:] for entry in entries if entry.startswith("?? ") and entry not in baseline_entries and entry[3:]]
    prohibited = [path for path in paths if _is_prohibited_commit_path(path)]
    if prohibited:
        return ExecutorStepResult(step.executor_id, "failed", "Setup generated possible credential files; exclusions refused", {"prohibited_paths": prohibited})
    if not paths:
        return ExecutorStepResult(step.executor_id, "passed", "setup generated no untracked files", {"excluded_paths": []})
    exclude_file = workdir / ".git" / "info" / "exclude"
    exclude_file.parent.mkdir(parents=True, exist_ok=True)
    existing = exclude_file.read_text(encoding="utf-8") if exclude_file.exists() else ""
    additions = [path for path in paths if path not in existing.splitlines()]
    if additions:
        with exclude_file.open("a", encoding="utf-8", newline="\n") as stream:
            if existing and not existing.endswith("\n"):
                stream.write("\n")
            stream.write("# SICKR Setup-generated workspace artifacts\n")
            stream.write("\n".join(additions) + "\n")
    return ExecutorStepResult(step.executor_id, "passed", "captured setup-generated local ignores", {"excluded_paths": paths, "exclude_file": str(exclude_file)})


def _porcelain_paths_z(output: str) -> list[str]:
    """Parse porcelain-v1 -z, including the second pathname for renames/copies."""
    fields = [field for field in output.split("\0") if field]
    paths: list[str] = []
    index = 0
    while index < len(fields):
        field = fields[index]
        if len(field) < 4:
            raise ValueError("malformed git status record")
        status, path = field[:2], field[3:]
        paths.append(path)
        index += 1
        if "R" in status or "C" in status:
            if index >= len(fields):
                raise ValueError("malformed git rename record")
            paths.append(fields[index])
            index += 1
    return paths


def _staged_paths_z(output: str) -> list[str]:
    fields = [field for field in output.split("\0") if field]
    paths: list[str] = []
    index = 0
    while index < len(fields):
        status = fields[index]
        index += 1
        count = 2 if status.startswith(("R", "C")) else 1
        if index + count > len(fields):
            raise ValueError("malformed staged name-status record")
        paths.extend(fields[index:index + count])
        index += count
    return paths


def _unsafe_commit_paths(workdir: Path, paths: list[str]) -> list[str]:
    unsafe: list[str] = []
    root = workdir.resolve()
    for path in dict.fromkeys(paths):
        if _is_prohibited_commit_path(path):
            unsafe.append(path)
            continue
        candidate = (workdir / path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            unsafe.append(path)
            continue
        if not candidate.exists() or not candidate.is_file():
            continue
        try:
            if candidate.stat().st_size > 5 * 1024 * 1024:
                unsafe.append(path)
            elif _SECRET_CONTENT_RE.search(candidate.read_text(encoding="utf-8", errors="ignore")):
                unsafe.append(path)
        except OSError:
            unsafe.append(path)
    return unsafe


def _git_commit_state_changes(*, step: ExecutorStep, ctx: ExecutorContext, actor_result: dict[str, Any] | None) -> ExecutorStepResult:
    """Commit a completed state's meaningful changes before publication.

    This executor is deliberately postflight-only. Failed or interrupted main
    executions retain their dirty per-ticket workspace for an affinity retry;
    they are not converted into opaque recovery refs or incomplete commits.
    """
    workdir = _ci_working_directory(ctx, step.params.get("working_directory"))
    if workdir is None:
        return ExecutorStepResult(step.executor_id, "error", "git.commit_state_changes working_directory must stay inside the workspace")
    if not workdir.exists():
        return ExecutorStepResult(step.executor_id, "error", "git.commit_state_changes working directory does not exist", {"working_directory": str(workdir)})
    timeout = _bounded_timeout(step.timeout_seconds) or 120
    inside = _git_step(["rev-parse", "--is-inside-work-tree"], cwd=workdir, env=ctx.env, timeout=timeout)
    if inside.returncode != 0 or inside.stdout.strip() != "true":
        return ExecutorStepResult(step.executor_id, "error", "git.commit_state_changes working directory is not a git checkout", inside.evidence("git_probe"))
    status = _git_step(["status", "--porcelain=v1", "-z", "--untracked-files=all"], cwd=workdir, env=ctx.env, timeout=timeout)
    if status.returncode != 0:
        return ExecutorStepResult(step.executor_id, "error", "git.commit_state_changes could not inspect worktree status", status.evidence("status"))
    if not status.stdout.strip():
        head = _git_step(["rev-parse", "HEAD"], cwd=workdir, env=ctx.env, timeout=timeout)
        return ExecutorStepResult(step.executor_id, "passed", "state changes already committed", {"committed": False, "head_sha": head.stdout.strip() if head.returncode == 0 else ""})

    try:
        changed_paths = _porcelain_paths_z(status.stdout)
    except ValueError as err:
        return ExecutorStepResult(step.executor_id, "error", f"git.commit_state_changes could not parse worktree status: {err}")
    prohibited = _unsafe_commit_paths(workdir, changed_paths)
    if prohibited:
        _git_step(["reset", "--mixed"], cwd=workdir, env=ctx.env, timeout=timeout)
        return ExecutorStepResult(
            step.executor_id,
            "failed",
            "git.commit_state_changes refused prohibited credential paths",
            {"prohibited_paths": prohibited},
        )

    added = _git_step(["add", "--all"], cwd=workdir, env=ctx.env, timeout=timeout)
    if added.returncode != 0:
        return ExecutorStepResult(step.executor_id, "failed", "git.commit_state_changes could not stage changes", added.evidence("git_add"))
    staged = _git_step(["diff", "--cached", "--name-status", "-z"], cwd=workdir, env=ctx.env, timeout=timeout)
    if staged.returncode != 0:
        _git_step(["reset", "--mixed"], cwd=workdir, env=ctx.env, timeout=timeout)
        return ExecutorStepResult(step.executor_id, "error", "git.commit_state_changes could not inspect staged changes", staged.evidence("staged_status"))
    try:
        staged_paths = _staged_paths_z(staged.stdout)
    except ValueError as err:
        _git_step(["reset", "--mixed"], cwd=workdir, env=ctx.env, timeout=timeout)
        return ExecutorStepResult(step.executor_id, "error", f"git.commit_state_changes could not parse staged changes: {err}")
    staged_prohibited = _unsafe_commit_paths(workdir, staged_paths)
    if staged_prohibited:
        _git_step(["reset", "--mixed"], cwd=workdir, env=ctx.env, timeout=timeout)
        return ExecutorStepResult(step.executor_id, "failed", "git.commit_state_changes refused prohibited or unscannable staged content", {"prohibited_paths": staged_prohibited})

    ticket_id = str(ctx.ticket.get("id") or "unknown")
    state_id = str(ctx.ticket.get("state_id") or ctx.ticket.get("current_state_id") or "unknown")
    summary = ""
    if isinstance(actor_result, dict):
        evidence = actor_result.get("evidence")
        if isinstance(evidence, dict):
            summary = str(evidence.get("summary") or "").strip()
    subject = str(step.params.get("message") or summary or f"chore(sickr): complete {state_id}").splitlines()[0][:200]
    message = "\n\n".join((
        subject,
        f"Sickr-Ticket: {ticket_id}",
        f"Sickr-State: {state_id}",
        f"Sickr-Lease: {ctx.lease_id or 'unknown'}",
        f"Sickr-Agent: {ctx.agent_id}",
    ))
    committed = _git_step(["commit", "-m", message], cwd=workdir, env=ctx.env, timeout=timeout)
    if committed.returncode != 0:
        _git_step(["reset", "--mixed"], cwd=workdir, env=ctx.env, timeout=timeout)
        return ExecutorStepResult(step.executor_id, "failed", "git.commit_state_changes could not commit staged changes", committed.evidence("git_commit"))
    head = _git_step(["rev-parse", "HEAD"], cwd=workdir, env=ctx.env, timeout=timeout)
    return ExecutorStepResult(step.executor_id, "passed", "state changes committed", {
        "committed": True,
        "head_sha": head.stdout.strip() if head.returncode == 0 else "",
        "changed_status": _truncate(status.stdout, 8192),
    })


def _github_publish_and_ensure_pr(*, step: ExecutorStep, ctx: ExecutorContext, actor_result: dict[str, Any] | None) -> ExecutorStepResult:
    base_branch, target_branch = _ticket_branch_pair(step=step, ctx=ctx)
    expected_repo = _ticket_repo_slug(ctx.ticket)
    evidence: dict[str, Any] = {
        "base_branch": base_branch,
        "target_branch": target_branch,
        "branch": target_branch,
    }
    if expected_repo:
        evidence["repo"] = expected_repo
    if not target_branch:
        return ExecutorStepResult(step.executor_id, "failed", "github.publish_and_ensure_pr requires a ticket branch", evidence)
    if target_branch == base_branch:
        return ExecutorStepResult(step.executor_id, "failed", "github.publish_and_ensure_pr refuses to publish the base branch", evidence)
    if not expected_repo:
        return ExecutorStepResult(step.executor_id, "failed", "github.publish_and_ensure_pr requires a GitHub repo slug", evidence)
    workdir = _ci_working_directory(ctx, step.params.get("working_directory"))
    if workdir is None:
        return ExecutorStepResult(step.executor_id, "error", "github.publish_and_ensure_pr working_directory must stay inside the workspace")
    if not workdir.exists() or not workdir.is_dir():
        return ExecutorStepResult(step.executor_id, "error", "github.publish_and_ensure_pr working directory does not exist", evidence)
    evidence["working_directory"] = str(workdir)
    timeout = step.timeout_seconds or 300

    current = _git_step(["branch", "--show-current"], cwd=workdir, env=ctx.env, timeout=timeout)
    if current.returncode != 0 or current.stdout.strip() != target_branch:
        return ExecutorStepResult(step.executor_id, "failed", "github.publish_and_ensure_pr must run on the ticket branch checkout", {**evidence, **current.evidence("current_branch")})
    status = _git_step(["status", "--porcelain"], cwd=workdir, env=ctx.env, timeout=timeout)
    if status.returncode != 0:
        return ExecutorStepResult(step.executor_id, "error", "github.publish_and_ensure_pr could not inspect worktree status", {**evidence, **status.evidence("status")})
    if status.stdout.strip():
        return ExecutorStepResult(step.executor_id, "failed", "github.publish_and_ensure_pr requires a clean worktree: " + _dirty_worktree_summary(status.stdout), {**evidence, "dirty_status": _truncate(status.stdout, 8192)})
    remote = _git_step(["remote", "get-url", "origin"], cwd=workdir, env=ctx.env, timeout=timeout)
    actual_repo = github_repo_from_remote(remote.stdout) if remote.returncode == 0 else None
    if actual_repo is None or actual_repo.lower() != expected_repo.lower():
        return ExecutorStepResult(step.executor_id, "failed", "github.publish_and_ensure_pr origin does not match the ticket repository", {**evidence, **remote.evidence("origin")})
    if not ctx.lease_id or ctx.workflow_client is None or not hasattr(ctx.workflow_client, "github_push_credential"):
        return ExecutorStepResult(step.executor_id, "error", "github.publish_and_ensure_pr requires an active SICKR lease")

    try:
        response = ctx.workflow_client.github_push_credential(ctx.lease_id, agent_id=ctx.agent_id)
    except AgentWorkflowClientError as err:
        return ExecutorStepResult(step.executor_id, "error", f"github.publish_and_ensure_pr credential request failed: {err}", {**evidence, "status": err.status})
    credential = response.get("credential") if isinstance(response, dict) else None
    if not isinstance(credential, dict):
        return ExecutorStepResult(step.executor_id, "error", "github.publish_and_ensure_pr received an invalid credential response", evidence)
    token = credential.get("token")
    repo = credential.get("repo")
    credential_base = credential.get("base_branch")
    credential_head = credential.get("head_branch")
    if not all(isinstance(value, str) and value.strip() for value in (token, repo, credential_base, credential_head)):
        return ExecutorStepResult(step.executor_id, "error", "github.publish_and_ensure_pr received an incomplete credential response", evidence)
    if repo.lower() != expected_repo.lower() or credential_base != base_branch or credential_head != target_branch:
        return ExecutorStepResult(step.executor_id, "error", "github.publish_and_ensure_pr credential scope does not match the ticket", evidence)

    network_env = github_token_env(token=token, base_env=_controlled_env(ctx.env))
    fetched = _git_step(["fetch", "--no-tags", "origin", base_branch], cwd=workdir, env=network_env, timeout=timeout)
    if fetched.returncode != 0:
        return ExecutorStepResult(step.executor_id, "failed", "github.publish_and_ensure_pr could not refresh the configured base branch", {**evidence, **fetched.evidence("base_fetch")})
    contains_base = _git_step(["merge-base", "--is-ancestor", f"origin/{base_branch}", "HEAD"], cwd=workdir, env=ctx.env, timeout=timeout)
    if contains_base.returncode != 0:
        return ExecutorStepResult(step.executor_id, "failed", f"github.publish_and_ensure_pr requires the ticket branch to include current origin/{base_branch}; reconcile retained work before publishing", {**evidence, **contains_base.evidence("base_ancestry")})

    head = _git_step(["rev-parse", "HEAD"], cwd=workdir, env=ctx.env, timeout=timeout)
    head_sha = head.stdout.strip() if head.returncode == 0 else ""
    push_preconditions = {
        "clean_worktree": True,
        "on_ticket_branch": True,
        # The push targets the exact leased branch with a scope-checked
        # credential; the branch either fast-forwards or the push is refused
        # by the remote (publish_ticket_branch never forces).
        "remote_fast_forward_or_absent": True,
        "current_base_included": True,
    }
    push_decision = authorize_completion(
        "git.push_unpushed_branch",
        evidence={"target_branch": target_branch, "head_sha": head_sha},
        preconditions=push_preconditions,
    )
    if not push_decision.allowed:
        return ExecutorStepResult(
            step.executor_id,
            "error",
            f"completion action denied: {push_decision.reason}",
            {**evidence, "completion_denied": push_decision.as_dict()},
        )
    try:
        push = publish_ticket_branch(
            workspace=workdir,
            repo=repo,
            branch=target_branch,
            token=token,
            base_env=_controlled_env(ctx.env),
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return ExecutorStepResult(step.executor_id, "failed", "github.publish_and_ensure_pr timed out while publishing the ticket branch", evidence)
    except OSError as err:
        return ExecutorStepResult(step.executor_id, "error", f"github.publish_and_ensure_pr could not publish the ticket branch: {err}", evidence)
    if push.returncode != 0:
        return ExecutorStepResult(step.executor_id, "failed", "github.publish_and_ensure_pr could not publish the ticket branch", {**evidence, "push_stderr": _truncate(push.stderr or push.stdout or "", 8192)})

    # Contract B: ensure-pr is reconciliation, not a blind mutation retry —
    # the runtime reads the exact (repo, base, head) state and returns the
    # existing PR when the earlier side effect landed before a timeout.
    reconcile_preconditions = {
        "branch_published": True,
        "exact_tuple_submitted": True,
    }
    reconcile_decision = authorize_completion(
        "pr.publish_reconcile",
        evidence={"repo": expected_repo, "base_branch": base_branch, "head_branch": target_branch},
        preconditions=reconcile_preconditions,
    )
    if not reconcile_decision.allowed:
        return ExecutorStepResult(
            step.executor_id,
            "error",
            f"completion action denied: {reconcile_decision.reason}",
            {**evidence, "published": True, "completion_denied": reconcile_decision.as_dict()},
        )
    pr_result = _ensure_pr_via_sickr_runtime(
        step=step,
        ctx=ctx,
        repo=expected_repo,
        base_branch=base_branch,
        target_branch=target_branch,
    )
    if pr_result is None:
        return ExecutorStepResult(step.executor_id, "error", "github.publish_and_ensure_pr requires the SICKR GitHub runtime", {**evidence, "published": True})
    audit = _pr_reconcile_audit(
        pr_result=pr_result,
        repo=expected_repo,
        base_branch=base_branch,
        target_branch=target_branch,
        head_sha=head_sha,
        preconditions_checked={**push_preconditions, **reconcile_preconditions},
    )
    return ExecutorStepResult(
        step.executor_id,
        pr_result.status,
        pr_result.message,
        {**evidence, "published": True, "completion_audit": audit, **pr_result.evidence},
    )


def _pr_reconcile_audit(
    *,
    pr_result: ExecutorStepResult,
    repo: str,
    base_branch: str,
    target_branch: str,
    head_sha: str,
    preconditions_checked: dict[str, Any],
) -> dict[str, Any]:
    """Audit the reconcile with what was OBSERVED, not what was intended.

    A failed ensure, an error, and an idempotent "PR already existed" must all
    be distinguishable from a fresh successful creation in the trail. The
    prior-PR state is derived from the runtime's own evidence when it says so
    (``created``), and recorded as unknown when it does not — this step never
    read the PR before the call, and the audit must not pretend it did.
    """
    created = pr_result.evidence.get("created")
    before: dict[str, Any] = {
        "pr_existed": (not created) if isinstance(created, bool) else None,
    }
    after: dict[str, Any] = {"ensure_pr_status": pr_result.status}
    for key in ("pr_url", "url", "html_url", "number", "created", "state", "isDraft"):
        if key in pr_result.evidence:
            after[key] = pr_result.evidence[key]
    nested_pr = pr_result.evidence.get("pr")
    if isinstance(nested_pr, dict):
        after.setdefault("pr", nested_pr)
    return completion_audit_record(
        action_id="pr.publish_reconcile",
        outcome="completed" if pr_result.status == "passed" else "failed",
        before=before,
        after=after,
        evidence_inputs={
            "repo": repo,
            "base_branch": base_branch,
            "head_branch": target_branch,
            "head_sha": head_sha,
        },
        preconditions_checked=preconditions_checked,
        idempotency_key=f"{repo}:{base_branch}:{target_branch}",
    )


def _successful_prior_evidence_roots(ctx: ExecutorContext) -> list[tuple[str, dict[str, Any]]]:
    roots: list[tuple[str, dict[str, Any]]] = []
    for index in range(len(ctx.prior_step_results) - 1, -1, -1):
        result = ctx.prior_step_results[index]
        evidence = result.get("evidence")
        if result.get("status") != "passed" or not isinstance(evidence, dict):
            continue
        executor_id = str(result.get("executor_id") or "unknown")
        roots.append((f"prior_step:{index}:{executor_id}", {"evidence": evidence}))
    return roots


def _ensure_pr_via_sickr_runtime(
    *,
    step: ExecutorStep,
    ctx: ExecutorContext,
    repo: str,
    base_branch: str,
    target_branch: str,
) -> ExecutorStepResult | None:
    client = ctx.workflow_client
    if not ctx.lease_id or client is None or not hasattr(client, "ensure_pull_request"):
        return None
    try:
        result = client.ensure_pull_request(
            ctx.lease_id,
            agent_id=ctx.agent_id,
            repo=repo,
            base_branch=base_branch,
            head_branch=target_branch,
            title=_pr_title(step=step, ctx=ctx),
            body=_pr_body(step=step, ctx=ctx),
        )
    except AgentWorkflowClientError as err:
        if err.status in {404, 501}:
            return None
        return ExecutorStepResult(step.executor_id, "error", f"{step.executor_id} service call failed: {err}", {"repo": repo, "base_branch": base_branch, "target_branch": target_branch, "status": err.status})
    except Exception:  # noqa: BLE001 - local-only/dev path when runtime client is unavailable before dispatch
        return None

    status = str(result.get("status") or "")
    message = str(result.get("message") or "")
    evidence = result.get("evidence") if isinstance(result.get("evidence"), dict) else {}
    if status == "passed":
        return ExecutorStepResult(step.executor_id, "passed", message or "pull request ensured by SICKR GitHub connection", evidence)
    if status == "failed":
        return ExecutorStepResult(step.executor_id, "failed", message or "SICKR GitHub connection could not ensure pull request", evidence)
    return ExecutorStepResult(step.executor_id, "error", message or "SICKR GitHub connection returned an invalid ensure-pr result", evidence)


def _github_sync_review_decision(*, step: ExecutorStep, ctx: ExecutorContext, actor_result: dict[str, Any] | None) -> ExecutorStepResult:
    decision = _extract_review_decision(step=step, ctx=ctx, actor_result=actor_result)
    if decision is None:
        return ExecutorStepResult(step.executor_id, "skipped", "review decision is missing")

    pr = _extract_pr_reference(ctx=ctx, actor_result=actor_result)
    if pr is None:
        return ExecutorStepResult(step.executor_id, "failed", "pull request reference is missing")

    client = ctx.workflow_client
    if not ctx.lease_id or client is None or not hasattr(client, "sync_github_review_decision"):
        return ExecutorStepResult(step.executor_id, "error", "github.sync_review_decision requires SICKR runtime GitHub connection", {"pr": pr, "decision": decision})

    body = _review_decision_body(step=step, ctx=ctx, actor_result=actor_result, decision=decision)
    try:
        result = client.sync_github_review_decision(
            ctx.lease_id,
            agent_id=ctx.agent_id,
            repo=str(pr["repo"]),
            pr_number=int(pr["number"]),
            decision=decision,
            body=body,
        )
    except AgentWorkflowClientError as err:
        return ExecutorStepResult(step.executor_id, "error", f"github.sync_review_decision service call failed: {err}", {"pr": pr, "decision": decision, "status": err.status})

    status = str(result.get("status") or "")
    message = str(result.get("message") or "")
    evidence = result.get("evidence") if isinstance(result.get("evidence"), dict) else {}
    evidence = {"pr": pr, "decision": decision, **evidence}
    if status == "passed":
        return ExecutorStepResult(step.executor_id, "passed", message or "GitHub review decision synced", evidence)
    if status == "skipped":
        return ExecutorStepResult(step.executor_id, "skipped", message or "GitHub review decision sync skipped", evidence)
    if status == "failed":
        return ExecutorStepResult(step.executor_id, "failed", message or "GitHub review decision sync failed", evidence)
    return ExecutorStepResult(step.executor_id, "error", message or "SICKR GitHub connection returned an invalid review-sync result", evidence)


def _github_prepare_pr_workspace(*, step: ExecutorStep, ctx: ExecutorContext, actor_result: dict[str, Any] | None) -> ExecutorStepResult:
    pr = _extract_pr_reference(ctx=ctx, actor_result=actor_result)
    if pr is None:
        return ExecutorStepResult(step.executor_id, "failed", "github.prepare_pr_workspace requires a pull request reference")

    expected_repo = _ticket_repo_slug(ctx.ticket)
    if expected_repo is None:
        return ExecutorStepResult(step.executor_id, "failed", "github.prepare_pr_workspace requires a GitHub ticket repository", {"pr": pr})
    if str(pr["repo"]).lower() != expected_repo.lower():
        return ExecutorStepResult(
            step.executor_id,
            "failed",
            "github.prepare_pr_workspace pull request does not match the ticket repository",
            {"pr": pr, "ticket_repo": expected_repo},
        )

    workdir = _ci_working_directory(ctx, step.params.get("working_directory"))
    if workdir is None:
        return ExecutorStepResult(step.executor_id, "error", "github.prepare_pr_workspace working_directory must stay inside the workspace")
    if not workdir.exists() or not workdir.is_dir():
        return ExecutorStepResult(
            step.executor_id,
            "error",
            "github.prepare_pr_workspace working directory does not exist",
            {"working_directory": str(workdir), "pr": pr},
        )

    timeout = step.timeout_seconds or 300
    inside = _git_step(["rev-parse", "--is-inside-work-tree"], cwd=workdir, env=ctx.env, timeout=timeout)
    if inside.returncode != 0 or inside.stdout.strip() != "true":
        return ExecutorStepResult(step.executor_id, "error", "github.prepare_pr_workspace working directory is not a git checkout", inside.evidence("git_probe"))
    status = _git_step(["status", "--porcelain"], cwd=workdir, env=ctx.env, timeout=timeout)
    if status.returncode != 0:
        return ExecutorStepResult(step.executor_id, "error", "github.prepare_pr_workspace could not inspect worktree status", status.evidence("status"))
    if status.stdout.strip():
        return ExecutorStepResult(
            step.executor_id,
            "failed",
            "github.prepare_pr_workspace requires a clean worktree before checkout: "
            + _dirty_worktree_summary(status.stdout),
            {"dirty_status": _truncate(status.stdout, 8192), "working_directory": str(workdir), "pr": pr},
        )

    remote = _git_step(["remote", "get-url", "origin"], cwd=workdir, env=ctx.env, timeout=timeout)
    actual_repo = github_repo_from_remote(remote.stdout) if remote.returncode == 0 else None
    if actual_repo is None or actual_repo.lower() != expected_repo.lower():
        return ExecutorStepResult(
            step.executor_id,
            "failed",
            "github.prepare_pr_workspace origin does not match the ticket repository",
            {"ticket_repo": expected_repo, "working_directory": str(workdir), **remote.evidence("origin")},
        )

    auth_env, provider, credential_error = _github_read_environment(ctx=ctx, expected_repo=expected_repo)
    if credential_error is not None:
        return ExecutorStepResult(step.executor_id, "error", credential_error, {"repo": expected_repo, "pr": pr})

    try:
        metadata = _github_pr_view(
            pr=pr,
            ctx=ctx,
            env=auth_env,
            fields=_GITHUB_PR_WORKSPACE_FIELDS,
        )
    except Exception as err:  # noqa: BLE001 - external GitHub metadata fault
        return ExecutorStepResult(step.executor_id, "error", f"github.prepare_pr_workspace metadata lookup failed: {err}", {"repo": expected_repo, "pr": pr})

    base_branch = _branch_param(metadata.get("baseRefName"))
    head_branch = _branch_param(metadata.get("headRefName"))
    expected_head_sha = _git_object_id(metadata.get("headRefOid"))
    if base_branch is None or head_branch is None or expected_head_sha is None:
        return ExecutorStepResult(
            step.executor_id,
            "error",
            "github.prepare_pr_workspace received incomplete pull request refs",
            {"repo": expected_repo, "pr": pr},
        )

    base_ref = "refs/remotes/sickr-review/base"
    head_ref = f"refs/remotes/sickr-review/pr-{pr['number']}"
    fetch = _git_step(
        [
            "fetch",
            "--force",
            _github_repo_fetch_url(expected_repo),
            f"+refs/heads/{base_branch}:{base_ref}",
            f"+refs/pull/{pr['number']}/head:{head_ref}",
        ],
        cwd=workdir,
        env=auth_env,
        timeout=timeout,
    )
    if fetch.returncode != 0:
        return ExecutorStepResult(
            step.executor_id,
            "failed",
            "github.prepare_pr_workspace could not fetch the pull request refs",
            {"repo": expected_repo, "pr": pr, "failure_stage": "fetch", **fetch.evidence("fetch")},
        )

    base_sha_result = _git_step(["rev-parse", base_ref], cwd=workdir, env=auth_env, timeout=timeout)
    head_sha_result = _git_step(["rev-parse", head_ref], cwd=workdir, env=auth_env, timeout=timeout)
    if base_sha_result.returncode != 0 or head_sha_result.returncode != 0:
        return ExecutorStepResult(
            step.executor_id,
            "error",
            "github.prepare_pr_workspace could not resolve fetched pull request refs",
            {"repo": expected_repo, "pr": pr, **base_sha_result.evidence("base_ref"), **head_sha_result.evidence("head_ref")},
        )
    base_sha = base_sha_result.stdout.strip()
    head_sha = head_sha_result.stdout.strip()
    if head_sha.lower() != expected_head_sha.lower():
        return ExecutorStepResult(
            step.executor_id,
            "failed",
            "github.prepare_pr_workspace fetched head does not match GitHub metadata",
            {"repo": expected_repo, "pr": pr, "expected_head_sha": expected_head_sha, "fetched_head_sha": head_sha},
        )

    checkout = _git_step(["checkout", "--detach", head_sha], cwd=workdir, env=auth_env, timeout=timeout)
    if checkout.returncode != 0:
        return ExecutorStepResult(
            step.executor_id,
            "failed",
            "github.prepare_pr_workspace could not checkout the pull request head",
            {"repo": expected_repo, "pr": pr, "failure_stage": "checkout", **checkout.evidence("checkout")},
        )

    merge_base_result = _git_step(["merge-base", base_sha, head_sha], cwd=workdir, env=auth_env, timeout=timeout)
    if merge_base_result.returncode != 0:
        return ExecutorStepResult(
            step.executor_id,
            "failed",
            "github.prepare_pr_workspace could not find a merge base",
            {"repo": expected_repo, "pr": pr, "base_sha": base_sha, "head_sha": head_sha, **merge_base_result.evidence("merge_base")},
        )
    merge_base_sha = merge_base_result.stdout.strip()
    changed = _git_step(["diff", "--name-only", f"{merge_base_sha}..{head_sha}"], cwd=workdir, env=auth_env, timeout=timeout)
    if changed.returncode != 0:
        return ExecutorStepResult(
            step.executor_id,
            "error",
            "github.prepare_pr_workspace could not enumerate pull request changes",
            {"repo": expected_repo, "pr": pr, **changed.evidence("diff")},
        )

    ticket_branch = _branch_param(ctx.ticket.get("branch")) or _branch_param(ctx.ticket.get("ticket_branch")) or _branch_param(ctx.ticket.get("head_branch"))
    evidence: dict[str, Any] = {
        "provider": provider,
        "repo": expected_repo,
        "pr_url": pr["url"],
        "pr_number": int(pr["number"]),
        "pr_state": str(metadata.get("state") or ""),
        "is_draft": metadata.get("isDraft") is True,
        "base_branch": base_branch,
        "base_sha": base_sha,
        "head_branch": head_branch,
        "head_sha": head_sha,
        "merge_base_sha": merge_base_sha,
        "diff_range": f"{merge_base_sha}...{head_sha}",
        "changed_files": [line for line in changed.stdout.splitlines() if line][:500],
        "working_directory": str(workdir),
        "checkout_mode": "detached",
    }
    if ticket_branch is not None:
        evidence["ticket_branch"] = ticket_branch
        evidence["ticket_branch_matches_pr"] = ticket_branch == head_branch
    return ExecutorStepResult(step.executor_id, "passed", "pull request workspace is ready for review", evidence)


def _github_read_environment(*, ctx: ExecutorContext, expected_repo: str) -> tuple[dict[str, str], str, str | None]:
    try:
        resolved = resolve_github_credential_environment(
            client=ctx.workflow_client,
            lease_id=ctx.lease_id,
            agent_id=ctx.agent_id,
            expected_repo=expected_repo,
            access="read",
            base_env=_controlled_env(ctx.env),
        )
    except GitHubCredentialEnvironmentError as err:
        return {}, "github_app", str(err)
    return resolved.env, resolved.provider, None


# Deploy verification --------------------------------------------------------
#
# A merge can silently deploy nothing: on 2026-07-20 a merge to main produced no
# Cloudflare build at all - no failed check, no PR comment, nothing to notice -
# and the change sat undeployed for two days. Watching for the provider's check
# to appear cannot catch that, because an absent check is indistinguishable from
# one that has not arrived yet, and it shares the failure mode of the very
# integration it is watching.
#
# So this asserts positively instead: the deployed artifact states which commit
# it was built from, and this step waits until that matches the branch head.

_DEPLOY_SHA_RE = re.compile(r"^[0-9a-f]{7,40}$", re.IGNORECASE)


def _deploy_probe_commit(url: str, *, field: str, timeout: int) -> tuple[str | None, int, str | None]:
    """Fetch the health endpoint and read the deployed commit out of it.

    Returns (commit, http_status, error). `commit` is None when the endpoint did
    not answer with a usable value; the caller decides whether that is fatal.
    """
    request = urllib.request.Request(url, headers={"User-Agent": "sickr-orchestrator/deploy-check"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.getcode()
            body = response.read(65536).decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001 - any transport failure is just "not yet"
        code = getattr(exc, "code", None)
        return None, int(code) if isinstance(code, int) else 0, str(exc)
    try:
        payload = json.loads(body)
    except ValueError:
        return None, status, "health endpoint did not return JSON"
    if not isinstance(payload, dict):
        return None, status, "health endpoint did not return a JSON object"
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        return None, status, f"health endpoint has no `{field}`"
    return value.strip(), status, None


def _deploy_commit_matches(deployed: str, expected: str) -> bool:
    """True when the deployed commit is the expected one.

    Compared as prefixes so a short sha on either side still matches, and only
    when both look like shas - a placeholder such as "development" must never
    satisfy the check.
    """
    left = deployed.strip().casefold()
    right = expected.strip().casefold()
    if not _DEPLOY_SHA_RE.match(left) or not _DEPLOY_SHA_RE.match(right):
        return False
    shortest = min(len(left), len(right))
    return left[:shortest] == right[:shortest]


def _github_ensure_deployed(*, step: ExecutorStep, ctx: ExecutorContext, actor_result: dict[str, Any] | None) -> ExecutorStepResult:
    del actor_result
    health_url = _nonempty_param(step.params, "health_url")
    if not health_url:
        return ExecutorStepResult(step.executor_id, "error", "github.ensure_deployed requires a health_url param")
    parsed = urlparse(health_url)
    if parsed.scheme != "https" or not parsed.netloc:
        return ExecutorStepResult(
            step.executor_id,
            "error",
            "github.ensure_deployed health_url must be an https URL",
            {"health_url": health_url},
        )

    commit_field = _nonempty_param(step.params, "commit_field") or "commit"
    # Default to the ticket's base branch - the branch a merge lands on, which
    # is what gets deployed. Validated like every other branch input here so an
    # authored value cannot reach a refspec as raw text.
    base_branch, _target_branch = _ticket_branch_pair(step=step, ctx=ctx)
    branch = _branch_param(step.params.get("branch")) or base_branch
    if not branch:
        return ExecutorStepResult(step.executor_id, "error", "github.ensure_deployed could not resolve a deploy branch")
    poll_budget = _bounded_positive_int(step.params.get("poll_seconds"), default=600, minimum=30, maximum=3600)
    poll_interval = _bounded_positive_int(step.params.get("poll_interval_seconds"), default=20, minimum=5, maximum=300)
    probe_timeout = _bounded_positive_int(step.params.get("probe_timeout_seconds"), default=15, minimum=1, maximum=60)
    retrigger = step.params.get("retrigger") is not False
    git_timeout = _bounded_timeout(step.timeout_seconds) or 120

    workdir = _ci_working_directory(ctx, step.params.get("working_directory"))
    if workdir is None:
        return ExecutorStepResult(step.executor_id, "error", "github.ensure_deployed working_directory escapes the workspace")

    evidence: dict[str, Any] = {"health_url": health_url, "branch": branch, "commit_field": commit_field}

    network_env = _git_network_environment(step=step, ctx=ctx, access="write" if retrigger else "read", evidence=evidence)
    if isinstance(network_env, ExecutorStepResult):
        return network_env

    head, remote = _remote_branch_head(branch=branch, workdir=workdir, env=network_env, timeout=git_timeout)
    if head is None:
        status: StepStatus = "error" if _is_github_auth_error(remote.stderr) else "failed"
        return ExecutorStepResult(
            step.executor_id,
            status,
            f"github.ensure_deployed could not read origin/{branch}",
            {**evidence, **remote.evidence("ls_remote")},
        )
    evidence["expected_sha"] = head

    def current_head() -> str | None:
        refreshed, _result = _remote_branch_head(branch=branch, workdir=workdir, env=network_env, timeout=git_timeout)
        return refreshed

    poll = _await_deployed_commit(
        health_url=health_url,
        commit_field=commit_field,
        expected_sha=head,
        expected_provider=current_head,
        budget_seconds=poll_budget,
        interval_seconds=poll_interval,
        probe_timeout=probe_timeout,
        log_fn=ctx.log_fn,
    )
    _record_poll(evidence, poll, suffix="")

    if poll.matched:
        return ExecutorStepResult(step.executor_id, "passed", f"{branch} is deployed ({poll.deployed[:12]})", evidence)  # type: ignore[index]

    if not retrigger:
        return ExecutorStepResult(
            step.executor_id,
            "failed",
            f"{branch} is not deployed after {poll_budget}s",
            {**evidence, "retriggered": False},
        )

    # An empty push is the provider-documented way to force a build when one was
    # never created. Exactly one attempt: if a second window also fails, the
    # cause is not a missed trigger and a human should look.
    push = _push_empty_retrigger_commit(branch=branch, workdir=workdir, env=network_env, timeout=git_timeout)
    evidence.update(push.evidence("retrigger_push"))
    evidence["retriggered"] = push.returncode == 0
    if push.returncode != 0:
        return ExecutorStepResult(
            step.executor_id,
            "failed",
            f"{branch} is not deployed and the retrigger push failed",
            evidence,
        )

    after = current_head() or poll.expected
    evidence["expected_sha_after_retrigger"] = after
    poll = _await_deployed_commit(
        health_url=health_url,
        commit_field=commit_field,
        expected_sha=after,
        expected_provider=current_head,
        budget_seconds=poll_budget,
        interval_seconds=poll_interval,
        probe_timeout=probe_timeout,
        log_fn=ctx.log_fn,
    )
    _record_poll(evidence, poll, suffix="_after_retrigger")

    if poll.matched:
        return ExecutorStepResult(
            step.executor_id,
            "passed",
            f"{branch} deployed after one retrigger ({poll.deployed[:12]})",  # type: ignore[index]
            evidence,
        )
    return ExecutorStepResult(
        step.executor_id,
        "failed",
        f"{branch} is still not deployed after a retrigger; the build pipeline needs a human",
        evidence,
    )


def _record_poll(evidence: dict[str, Any], poll: DeployPollResult, *, suffix: str) -> None:
    evidence[f"poll_attempts{suffix}"] = poll.attempts
    if poll.deployed is not None:
        evidence["deployed_sha"] = poll.deployed
    if poll.expected:
        evidence[f"expected_sha_observed{suffix}"] = poll.expected
    if poll.last_error:
        evidence[f"last_probe_error{suffix}"] = _truncate(poll.last_error, 2048)


def _remote_branch_head(*, branch: str, workdir: Path, env: dict[str, str], timeout: int) -> tuple[str | None, GitStepResult]:
    result = _git_step(["ls-remote", "origin", f"refs/heads/{branch}"], cwd=workdir, env=env, timeout=timeout)
    if result.returncode != 0 or not result.stdout.strip():
        return None, result
    return result.stdout.split()[0].strip(), result


def _await_deployed_commit(
    *,
    health_url: str,
    commit_field: str,
    expected_sha: str,
    expected_provider: Any,
    budget_seconds: int,
    interval_seconds: int,
    probe_timeout: int,
    log_fn: Any,
) -> DeployPollResult:
    """Poll until the deployed commit matches the branch head, or time runs out.

    The expected sha is re-read between probes rather than snapshotted. Two
    merges landing minutes apart is routine here, and the provider coalesces
    builds: production can legitimately jump straight to the newer commit and
    never serve the one this step started from. Holding the original sha would
    then escalate - or push a junk retrigger commit - on a ticket that deployed
    perfectly well.
    """
    deadline = time.monotonic() + budget_seconds
    expected = expected_sha
    attempts = 0
    # Retain the last sha production actually reported: a transport blip on the
    # final probe would otherwise erase the one detail a human reading the
    # escalation needs.
    last_seen: str | None = None
    last_error: str | None = None
    while True:
        attempts += 1
        probe, _status, error = _deploy_probe_commit(health_url, field=commit_field, timeout=probe_timeout)
        if probe is not None:
            last_seen = probe
        else:
            last_error = error
        if probe is not None and _deploy_commit_matches(probe, expected):
            return DeployPollResult(True, probe, expected, attempts, None)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return DeployPollResult(False, last_seen, expected, attempts, last_error)
        if callable(log_fn):
            log_fn(
                f"deploy check: serving {last_seen or 'unknown'}, waiting for {expected[:12]} "
                f"({int(remaining)}s left)"
            )
        time.sleep(min(interval_seconds, max(remaining, 0)))
        refreshed = expected_provider() if callable(expected_provider) else None
        if refreshed:
            expected = refreshed


def _push_empty_retrigger_commit(*, branch: str, workdir: Path, env: dict[str, str], timeout: int) -> GitStepResult:
    """Push one empty commit to `branch` to force a build.

    Uses a detached fetch/commit-tree flow rather than touching the checkout, so
    a ticket workspace on another branch is never disturbed. The push is a plain
    fast-forward - never forced - so if the branch moved underneath us git
    rejects it and the step fails rather than clobbering anyone.
    """
    fetched = _git_step(["fetch", "origin", f"{branch}:refs/remotes/origin/{branch}"], cwd=workdir, env=env, timeout=timeout)
    if fetched.returncode != 0:
        return fetched
    head = _git_step(["rev-parse", f"origin/{branch}"], cwd=workdir, env=env, timeout=timeout)
    if head.returncode != 0:
        return head
    parent = head.stdout.strip()
    tree = _git_step(["rev-parse", f"{parent}^{{tree}}"], cwd=workdir, env=env, timeout=timeout)
    if tree.returncode != 0:
        return tree
    created = _git_step(
        [
            # Identity must be explicit: the hermetic env carries no
            # GIT_COMMITTER_*, so on a daemon host without a global user.email
            # commit-tree would abort and the one automatic remedy in this
            # design would never fire.
            "-c",
            "user.name=SICKR Workflow",
            "-c",
            "user.email=workflow@sickr.ai",
            "commit-tree",
            tree.stdout.strip(),
            "-p",
            parent,
            "-m",
            "chore(deploy): empty commit to retrigger a missed build [sickr]",
        ],
        cwd=workdir,
        env=env,
        timeout=timeout,
    )
    if created.returncode != 0:
        return created
    return _git_step(["push", "origin", f"{created.stdout.strip()}:refs/heads/{branch}"], cwd=workdir, env=env, timeout=timeout)


def _pr_extract_context(*, step: ExecutorStep, ctx: ExecutorContext, actor_result: dict[str, Any] | None) -> ExecutorStepResult:
    pr = _extract_pr_reference(ctx=ctx, actor_result=actor_result)
    if pr is None:
        if step.params.get("required") is True:
            return ExecutorStepResult(step.executor_id, "failed", "pull request reference is missing")
        return ExecutorStepResult(step.executor_id, "skipped", "pull request reference is missing")

    evidence: dict[str, Any] = {
        "pr": pr,
        "pr_url": pr["url"],
        "pr_number": str(pr["number"]),
        "pr_links": [pr["url"]],
    }
    github_env, _provider, credential_error = _github_read_environment(
        ctx=ctx,
        expected_repo=str(pr["repo"]),
    )
    if credential_error is not None:
        evidence["metadata_error"] = credential_error[:1024]
        return ExecutorStepResult(step.executor_id, "passed", "pull request context extracted without GitHub metadata", evidence)
    try:
        metadata = _github_pr_view(pr=pr, ctx=ctx, env=github_env)
    except Exception as err:  # noqa: BLE001 - enrichment should not block by default
        evidence["metadata_error"] = str(err)[:1024]
        return ExecutorStepResult(step.executor_id, "passed", "pull request context extracted without GitHub metadata", evidence)

    evidence["pr"] = {**pr, **metadata}
    author = metadata.get("author")
    if isinstance(author, dict) and isinstance(author.get("login"), str) and author["login"].strip():
        evidence["implementation_author"] = author["login"].strip()
    return ExecutorStepResult(step.executor_id, "passed", "pull request context extracted", evidence)


def _pr_approved(*, step: ExecutorStep, ctx: ExecutorContext, actor_result: dict[str, Any] | None) -> ExecutorStepResult:
    candidates = _ship_candidate_pr_urls(ctx)
    if len(candidates) <= 1:
        return _pr_approved_single(step=step, ctx=ctx, actor_result=actor_result)
    approvals: list[dict[str, Any]] = []
    for url in candidates:
        ticket = {**ctx.ticket, "pr_links": [url], "pr_url": url}
        # A preceding sync step describes one concrete PR. It must not be
        # replayed as approval for every candidate in a multi-PR shipment.
        candidate_ctx = replace(ctx, ticket=ticket, prior_step_results=())
        result = _pr_approved_single(step=step, ctx=candidate_ctx, actor_result=None)
        approvals.append({**result.as_dict(), "pr_url": url})
    passed = all(item.get("status") == "passed" for item in approvals)
    return ExecutorStepResult(
        step.executor_id,
        "passed" if passed else "failed",
        "all pull requests are approved" if passed else "one or more pull requests are not approved",
        {"approvals": approvals, "pr_links": candidates},
    )


def _ship_candidate_pr_urls(ctx: ExecutorContext) -> list[str]:
    raw = ctx.ticket.get("pr_links")
    if not isinstance(raw, list):
        return []
    return _dedupe_preserve([value.strip() for value in raw if isinstance(value, str) and value.strip()])


def _pr_approved_single(*, step: ExecutorStep, ctx: ExecutorContext, actor_result: dict[str, Any] | None) -> ExecutorStepResult:
    synced = _prior_synced_sickr_approval(ctx)
    pr = _extract_pr_reference(ctx=ctx, actor_result=actor_result)
    if synced is None and pr is not None:
        synced = _accumulated_sickr_approval(ctx=ctx, pr=pr)
    if synced is not None:
        formal = synced.get("formal_review") is True
        evidence: dict[str, Any] = {
            "approval_source": synced.get("approval_source") or "sickr_synced_review",
            "formal_github_approval": formal,
            "sickr_review_decision": "approve",
        }
        for key in (
            "pr",
            "pr_url",
            "pr_number",
            "delivery_mode",
            "github_review_event",
            "review_url",
            "github_limitation",
            "state_role_id",
            "evidence_ref",
        ):
            if key in synced:
                evidence[key] = synced[key]
        message = "pull request has a synchronized SICKR approval"
        if not formal:
            message += "; GitHub formal approval is unavailable because the GitHub App authored the pull request"
        return ExecutorStepResult(step.executor_id, "passed", message, evidence)

    loaded = _load_pr_metadata(
        executor_id=step.executor_id,
        ctx=ctx,
        actor_result=actor_result,
        fields=_GITHUB_PR_APPROVAL_FIELDS,
    )
    if isinstance(loaded, ExecutorStepResult):
        return loaded
    pr, metadata = loaded
    state = str(metadata.get("state") or "").upper()
    merged_at = metadata.get("mergedAt")
    review_decision = str(metadata.get("reviewDecision") or "").upper()
    evidence = {"pr": {**pr, **metadata}, "reviewDecision": review_decision}
    if state == "MERGED" or merged_at:
        evidence["already_merged"] = True
        evidence["governance_exceptions"] = [] if review_decision == "APPROVED" else ["approval_not_observable_after_merge"]
        return ExecutorStepResult(step.executor_id, "passed", "pull request is already merged", evidence)
    if review_decision == "APPROVED":
        return ExecutorStepResult(step.executor_id, "passed", "pull request is approved", evidence)
    return ExecutorStepResult(step.executor_id, "failed", "pull request is not approved", evidence)


def _prior_synced_sickr_approval(ctx: ExecutorContext) -> dict[str, Any] | None:
    if not ctx.prior_step_results:
        return None
    result = ctx.prior_step_results[-1]
    if result.get("executor_id") != "github.sync_review_decision" or result.get("status") != "passed":
        return None
    evidence = result.get("evidence")
    if not isinstance(evidence, dict):
        return None
    decision = str(evidence.get("sickr_review_decision") or "").strip().lower()
    return evidence if decision == "approve" else None


def _accumulated_sickr_approval(*, ctx: ExecutorContext, pr: dict[str, Any]) -> dict[str, Any] | None:
    """Return the latest canonical review decision for this exact PR.

    Workflow-service accumulation intentionally inlines bounded outputs, not
    heavy hook evidence. A review decision may be emitted separately from the
    implementation run that introduced the PR, so correlate them by canonical
    timeline while refusing to apply a decision to a PR introduced afterward.
    """
    candidates: list[tuple[str, int, str, dict[str, Any], str]] = []
    accumulated_runs: list[tuple[int, str, dict[str, Any], dict[str, Any], str]] = []
    sequence = 0
    for state_role_id, runs in ctx.states.items():
        if not isinstance(runs, list):
            continue
        for run in runs:
            sequence += 1
            if not isinstance(run, dict) or run.get("status") != "completed":
                continue
            outputs = run.get("outputs")
            if not isinstance(outputs, dict):
                continue
            timeline = str(run.get("ended_at") or run.get("started_at") or "")
            accumulated_runs.append((sequence, state_role_id, run, outputs, timeline))

    pr_observations = [
        timeline for _sequence, _state_role_id, _run, outputs, timeline in accumulated_runs
        if timeline and _outputs_reference_pr(outputs, pr)
    ]
    for sequence, state_role_id, run, outputs, timeline in accumulated_runs:
        decision = _review_decision_from_outputs(outputs)
        if decision is None:
            continue
        # Newer channel-native review runs may carry only their decision;
        # the PR artifact remains on the preceding implementation run.
        # Associate them only when this exact PR was observed no later than
        # the decision. This prevents an old approval from authorizing a
        # replacement PR introduced by a later implementation retry.
        references_pr = _outputs_reference_pr(outputs, pr)
        observed_before_decision = bool(timeline) and any(
            observed_at <= timeline for observed_at in pr_observations
        )
        if not references_pr and not observed_before_decision:
            continue
        candidates.append((timeline, sequence, state_role_id, run, decision))
    if not candidates:
        return None
    _timeline, _sequence, state_role_id, run, decision = max(candidates, key=lambda item: (item[0], item[1]))
    if decision != "approve" or run.get("result") != "success":
        return None
    return {
        "approval_source": "accumulated_state_output",
        "formal_review": False,
        "sickr_review_decision": "approve",
        "state_role_id": state_role_id,
        "evidence_ref": run.get("evidence_ref"),
        "pr": pr,
        "pr_url": pr["url"],
        "pr_number": pr["number"],
    }


def _outputs_reference_pr(outputs: dict[str, Any], expected_pr: dict[str, Any]) -> bool:
    pull_requests = outputs.get("pull_requests")
    if not isinstance(pull_requests, list):
        return False
    for value in pull_requests:
        parsed = _parse_pr_reference(value)
        if parsed is not None and parsed["repo"].lower() == str(expected_pr["repo"]).lower() and parsed["number"] == expected_pr["number"]:
            return True
    return False


def _review_decision_from_outputs(outputs: dict[str, Any]) -> str | None:
    decisions = outputs.get("decisions")
    if not isinstance(decisions, list):
        return None
    for item in reversed(decisions):
        if not isinstance(item, dict) or item.get("key") not in {"decision.review.outcome", "review_outcome", "review_decision"}:
            continue
        return _normalize_review_decision(item.get("value"))
    return None


def _pr_checks_green(*, step: ExecutorStep, ctx: ExecutorContext, actor_result: dict[str, Any] | None) -> ExecutorStepResult:
    timeout_seconds = _bounded_positive_int(step.params.get("wait_timeout_seconds"), default=600, minimum=0, maximum=1800)
    poll_interval_seconds = _bounded_positive_int(step.params.get("poll_interval_seconds"), default=10, minimum=1, maximum=60)
    deadline = time.monotonic() + timeout_seconds
    waited = False
    last_pending: list[Any] = []
    last_metadata: dict[str, Any] | None = None
    pr: dict[str, Any] | None = None

    while True:
        loaded = _load_pr_check_metadata(
            executor_id=step.executor_id,
            ctx=ctx,
            actor_result=actor_result,
        )
        if isinstance(loaded, ExecutorStepResult):
            return loaded
        pr, metadata = loaded
        last_metadata = metadata
        checks = metadata.get("statusCheckRollup")
        if not isinstance(checks, list) or len(checks) == 0:
            return ExecutorStepResult(step.executor_id, "passed", "pull request has no status checks", {"pr": {**pr, **metadata}, "statusCheckRollup": [], "waited_for_checks": waited, "verification": "confirmed"})
        else:
            pending = [check for check in checks if _check_is_pending(check)]
            failing = [check for check in checks if not _check_is_pending(check) and not _check_is_green(check)]
            evidence = {
                "pr": {**pr, **metadata},
                "statusCheckRollup": checks,
                "pending_checks": pending,
                "failing_checks": failing,
                "waited_for_checks": waited,
            }
            if failing:
                # The world was observed and contradicts the claim: mismatch.
                return ExecutorStepResult(step.executor_id, "failed", "pull request status checks are not green", {**evidence, "verification": "mismatch"})
            if not pending:
                return ExecutorStepResult(step.executor_id, "passed", "pull request status checks are green", {**evidence, "verification": "confirmed"})
            last_pending = pending

        if time.monotonic() >= deadline:
            evidence = {
                "pr": ({**pr, **last_metadata} if pr is not None and last_metadata is not None else {}),
                "statusCheckRollup": last_metadata.get("statusCheckRollup") if isinstance(last_metadata, dict) else [],
                "pending_checks": last_pending,
                "waited_for_checks": waited,
                "wait_timeout_seconds": timeout_seconds,
            }
            # Checks still pending at the deadline proved neither success nor
            # failure. Inconclusive: the run parks for re-verification instead
            # of declaring the implementation failed and rerunning Main.
            message = "pull request status checks are still pending" if last_pending else "pull request has no status checks"
            return _inconclusive(step.executor_id, message, evidence)

        waited = True
        time.sleep(min(poll_interval_seconds, max(0.0, deadline - time.monotonic())))


def _pr_mergeable(*, step: ExecutorStep, ctx: ExecutorContext, actor_result: dict[str, Any] | None) -> ExecutorStepResult:
    loaded = _load_pr_metadata(
        executor_id=step.executor_id,
        ctx=ctx,
        actor_result=actor_result,
        fields=_GITHUB_PR_MERGEABILITY_FIELDS,
    )
    if isinstance(loaded, ExecutorStepResult):
        return loaded
    pr, metadata = loaded
    state = str(metadata.get("state") or "").upper()
    merge_state = str(metadata.get("mergeStateStatus") or "").upper()
    is_draft = metadata.get("isDraft") is True
    evidence = {"pr": {**pr, **metadata}, "state": state, "mergeStateStatus": merge_state, "isDraft": is_draft}
    if state == "OPEN" and not is_draft and merge_state in {"CLEAN", "HAS_HOOKS"}:
        return ExecutorStepResult(step.executor_id, "passed", "pull request is mergeable", evidence)
    return ExecutorStepResult(step.executor_id, "failed", "pull request is not mergeable", evidence)


def _load_pr_metadata(
    *,
    executor_id: str,
    ctx: ExecutorContext,
    actor_result: dict[str, Any] | None,
    fields: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]] | ExecutorStepResult:
    pr = _extract_pr_reference(ctx=ctx, actor_result=actor_result)
    if pr is None:
        return ExecutorStepResult(executor_id, "failed", "pull request reference is missing")
    github_env, provider, credential_error = _github_read_environment(
        ctx=ctx,
        expected_repo=str(pr["repo"]),
    )
    if credential_error is not None:
        # Could not even attempt the observation — inconclusive, not a verdict.
        return _inconclusive(
            executor_id,
            credential_error,
            {"pr": pr, "credential_provider": provider},
        )
    try:
        return pr, _github_pr_view(
            pr=pr,
            ctx=ctx,
            env=github_env,
            fields=fields or _GITHUB_PR_VIEW_FIELDS,
        )
    except Exception as err:  # noqa: BLE001 - the world was not observed; inconclusive
        return _inconclusive(
            executor_id, f"pull request metadata lookup failed: {err}", {"pr": pr}
        )


def _load_pr_check_metadata(
    *,
    executor_id: str,
    ctx: ExecutorContext,
    actor_result: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]] | ExecutorStepResult:
    pr = _extract_pr_reference(ctx=ctx, actor_result=actor_result)
    if pr is None:
        return ExecutorStepResult(executor_id, "failed", "pull request reference is missing")
    github_env, provider, credential_error = _github_read_environment(
        ctx=ctx,
        expected_repo=str(pr["repo"]),
    )
    if credential_error is not None:
        return _inconclusive(
            executor_id,
            credential_error,
            {"pr": pr, "credential_provider": provider},
        )
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            metadata = _github_pr_view(
                pr=pr,
                ctx=ctx,
                env=github_env,
                fields=_GITHUB_PR_CHECK_FIELDS,
            )
            checks = metadata.get("statusCheckRollup")
            if not isinstance(checks, list):
                head_sha = metadata.get("headRefOid")
                if not isinstance(head_sha, str) or not head_sha.strip():
                    return ExecutorStepResult(
                        executor_id,
                        "error",
                        "pull request head commit is missing",
                        {"pr": {**pr, **metadata}},
                    )
                checks = read_commit_checks(
                    repo=str(pr["repo"]),
                    commit_sha=head_sha.strip(),
                    cwd=ctx.workspace_root if ctx.workspace_root.exists() else None,
                    env=_controlled_env(github_env),
                )
                metadata = {
                    **metadata,
                    "statusCheckRollup": checks,
                    "checks_source": "github_rest",
                }
            if attempt > 1:
                metadata = {**metadata, "checks_lookup_attempts": attempt}
            return pr, metadata
        except Exception as err:  # noqa: BLE001 - surfaced as executor system fault
            winerror = getattr(err, "winerror", None)
            transient = winerror in {8, 14, 1450, 1455}
            if transient and attempt < max_attempts:
                time.sleep(float(attempt))
                continue
            evidence: dict[str, Any] = {
                "pr": pr,
                "checks_lookup_attempts": attempt,
                "transient_resource_error": transient,
            }
            if isinstance(winerror, int):
                evidence["winerror"] = winerror
            return _inconclusive(
                executor_id,
                f"pull request checks lookup failed after {attempt} attempt(s): {err}",
                evidence,
            )


def _extract_pr_reference(*, ctx: ExecutorContext, actor_result: dict[str, Any] | None) -> dict[str, Any] | None:
    for value in _candidate_pr_values(ctx=ctx, actor_result=actor_result):
        pr = _parse_pr_reference(value)
        if pr is not None:
            return pr
    return None


def _extract_pr_reference_before_ticket(*, ctx: ExecutorContext, actor_result: dict[str, Any] | None) -> dict[str, Any] | None:
    prior_roots = [root.get("evidence") for _name, root in _successful_prior_evidence_roots(ctx)]
    for root in (*prior_roots, actor_result):
        if not isinstance(root, dict):
            continue
        for value in _pr_values_from_root(root):
            pr = _parse_pr_reference(value)
            if pr is not None:
                return pr
    return None


def _candidate_pr_values(*, ctx: ExecutorContext, actor_result: dict[str, Any] | None) -> list[Any]:
    values: list[Any] = []
    prior_roots = [root.get("evidence") for _name, root in _successful_prior_evidence_roots(ctx)]
    for root in (*prior_roots, actor_result, ctx.ticket):
        if not isinstance(root, dict):
            continue
        values.extend(_pr_values_from_root(root))
    return values


def _pr_values_from_root(root: dict[str, Any]) -> list[Any]:
    return [
        root.get("pr_url"),
        root.get("pull_request_url"),
        root.get("pr_link"),
        root.get("pr_links"),
        root.get("title"),
        root.get("description"),
        root.get("evidence"),
    ]


def _parse_pr_reference(value: Any) -> dict[str, Any] | None:
    if isinstance(value, str):
        match = _PR_URL_RE.search(value)
        if not match:
            return None
        owner = match.group("owner")
        repo_name = match.group("repo")
        number = int(match.group("number"))
        repo = f"{owner}/{repo_name}"
        return {
            "url": f"https://github.com/{repo}/pull/{number}",
            "repo": repo,
            "number": number,
        }
    if isinstance(value, list):
        for item in value:
            parsed = _parse_pr_reference(item)
            if parsed is not None:
                return parsed
    if isinstance(value, dict):
        for key in ("url", "html_url", "pr_url", "pull_request_url", "link"):
            parsed = _parse_pr_reference(value.get(key))
            if parsed is not None:
                return parsed
        for item in value.values():
            parsed = _parse_pr_reference(item)
            if parsed is not None:
                return parsed
    return None


def _extract_review_decision(*, step: ExecutorStep, ctx: ExecutorContext, actor_result: dict[str, Any] | None) -> str | None:
    channel = _nonempty_param(step.params, "channel")
    if channel:
        found, value, _root = _resolve_channel_value(channel=channel, ctx=ctx, actor_result=actor_result)
        return _normalize_review_decision(value) if found else None
    configured = _normalize_review_decision(step.params.get("decision"))
    if configured is not None:
        return configured
    for value in _candidate_review_decision_values(ctx=ctx, actor_result=actor_result):
        normalized = _normalize_review_decision(value)
        if normalized is not None:
            return normalized
    return None


def _candidate_review_decision_values(*, ctx: ExecutorContext, actor_result: dict[str, Any] | None) -> list[Any]:
    values: list[Any] = []
    for root in _candidate_review_roots(ctx=ctx, actor_result=actor_result):
        if not isinstance(root, dict):
            continue
        values.extend([
            root.get("review_outcome"),
            root.get("review_decision"),
            root.get("decision"),
            root.get("approved"),
        ])
        review = root.get("review")
        if isinstance(review, dict):
            values.extend([
                review.get("decision"),
                review.get("outcome"),
                review.get("approved"),
            ])
    return values


def _candidate_review_roots(*, ctx: ExecutorContext, actor_result: dict[str, Any] | None) -> list[Any]:
    roots: list[Any] = []
    prior_roots = [root.get("evidence") for _name, root in _successful_prior_evidence_roots(ctx)]
    for root in (*prior_roots, actor_result, ctx.ticket.get("evidence"), ctx.ticket):
        roots.append(root)
        if isinstance(root, dict) and isinstance(root.get("evidence"), dict):
            roots.append(root.get("evidence"))
    return roots


def _normalize_review_decision(value: Any) -> str | None:
    if isinstance(value, bool):
        return "approve" if value else "request_changes"
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"approve", "approved", "approval", "accept", "accepted", "true", "yes"}:
        return "approve"
    if normalized in {"request_changes", "changes_requested", "requested_changes", "reject", "rejected", "false", "no"}:
        return "request_changes"
    if normalized in {"comment", "neutral", "info", "informational"}:
        return "comment"
    return None


def _review_decision_body(*, step: ExecutorStep, ctx: ExecutorContext, actor_result: dict[str, Any] | None, decision: str) -> str:
    body = step.params.get("body")
    if isinstance(body, str) and body.strip():
        return body.strip()
    for value in _candidate_review_body_values(ctx=ctx, actor_result=actor_result):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return f"SICKR review decision: {decision.replace('_', ' ')}."


def _candidate_review_body_values(*, ctx: ExecutorContext, actor_result: dict[str, Any] | None) -> list[Any]:
    values: list[Any] = []
    for root in _candidate_review_roots(ctx=ctx, actor_result=actor_result):
        if not isinstance(root, dict):
            continue
        values.extend([
            root.get("review_summary"),
            root.get("review_comments"),
            root.get("summary"),
            root.get("message"),
            root.get("comment"),
        ])
        review = root.get("review")
        if isinstance(review, dict):
            values.extend([
                review.get("summary"),
                review.get("comments"),
                review.get("body"),
                review.get("message"),
            ])
    return values


_GITHUB_PR_VIEW_FIELDS = (
    "url,number,title,state,isDraft,mergeStateStatus,reviewDecision,author,"
    "headRefName,headRefOid,baseRefName,"
    "isCrossRepository,headRepository,headRepositoryOwner"
)
_GITHUB_PR_WORKSPACE_FIELDS = "url,number,state,isDraft,headRefName,headRefOid,baseRefName"
_GITHUB_PR_APPROVAL_FIELDS = "url,number,state,mergedAt,mergeCommit,reviewDecision"
_GITHUB_PR_CHECK_FIELDS = "url,number,headRefOid"
_GITHUB_PR_MERGEABILITY_FIELDS = "url,number,state,isDraft,mergeStateStatus"


def _github_pr_view(
    *,
    pr: dict[str, Any],
    ctx: ExecutorContext,
    env: dict[str, str] | None = None,
    fields: str = _GITHUB_PR_VIEW_FIELDS,
) -> dict[str, Any]:
    proc = run_process(
        [
            "gh",
            "pr",
            "view",
            str(pr["number"]),
            "--repo",
            str(pr["repo"]),
            "--json",
            fields,
        ],
        cwd=ctx.workspace_root if ctx.workspace_root.exists() else None,
        env=_controlled_env(env or ctx.env),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(_truncate(proc.stderr or proc.stdout or f"gh exited {proc.returncode}", 2048))
    parsed = json.loads(proc.stdout or "{}")
    if not isinstance(parsed, dict):
        raise RuntimeError("gh returned non-object PR metadata")
    return parsed


def _ticket_branch_pair(*, step: ExecutorStep, ctx: ExecutorContext) -> tuple[str, str | None]:
    base_branch = _branch_param(step.params.get("base_branch")) or _branch_param(ctx.ticket.get("base_branch")) or "main"
    target_branch = (
        _branch_param(step.params.get("target_branch"))
        or _branch_param(ctx.ticket.get("branch"))
        or _branch_param(ctx.ticket.get("ticket_branch"))
        or _branch_param(ctx.ticket.get("head_branch"))
    )
    return base_branch, target_branch


def _uses_authoritative_validation_revisions(ctx: ExecutorContext) -> bool:
    if str(ctx.ticket.get("governance_state_type") or "").strip().lower() == "evaluation":
        return True
    # Compatibility for workflow-service versions that materialize a
    # context-only authoritative workspace but omit governance_state_type from
    # ticket_context. The bootstrap variable is generated by SICKR itself and
    # is therefore a stronger signal than generic action/role labels.
    return bool(str(ctx.env.get("SICKR_AUTHORITATIVE_REPOS") or "").strip())


def _authoritative_validation_preflight_result(
    *, step: ExecutorStep, ctx: ExecutorContext
) -> ExecutorStepResult:
    raw = ctx.env.get("SICKR_AUTHORITATIVE_REPOS", "")
    try:
        repositories = json.loads(raw) if raw else []
    except json.JSONDecodeError:
        repositories = []
    if not isinstance(repositories, list) or not repositories:
        return ExecutorStepResult(
            step.executor_id,
            "error",
            "authoritative validation repository evidence is missing from workspace bootstrap",
            {"workspace_mode": "authoritative_validation"},
        )
    failed = [
        record
        for record in repositories
        if not isinstance(record, dict)
        or record.get("synced") is not True
        or not str(record.get("head_sha") or "").strip()
    ]
    if failed:
        return ExecutorStepResult(
            step.executor_id,
            "failed",
            "authoritative validation repositories are not fully synchronized",
            {"workspace_mode": "authoritative_validation", "repositories": repositories},
        )
    return ExecutorStepResult(
        step.executor_id,
        "passed",
        "validation uses synchronized upstream default branches; a ticket branch is not required",
        {"workspace_mode": "authoritative_validation", "repositories": repositories},
    )


def _github_repo_fetch_url(repo: str) -> str:
    return f"https://github.com/{repo}.git"


def _git_object_id(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    object_id = value.strip()
    return object_id if re.fullmatch(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", object_id) else None


def _ticket_repo_slug(ticket: dict[str, Any]) -> str | None:
    for key in ("repo", "repository", "repo_slug", "repository_slug", "repo_url", "repository_url"):
        value = ticket.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        candidate = value.strip().rstrip("/")
        if candidate.endswith(".git"):
            candidate = candidate[:-4]
        if "github.com/" in candidate:
            candidate = candidate.split("github.com/", 1)[1]
        if candidate.startswith("git@github.com:"):
            candidate = candidate.removeprefix("git@github.com:")
        candidate = candidate.strip("/")
        parts = candidate.split("/")
        if len(parts) >= 2:
            slug = f"{parts[-2]}/{parts[-1]}"
            if re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", slug):
                return slug
    return None


def _is_github_auth_error(stderr: str) -> bool:
    text = str(stderr or "").lower()
    return (
        "gh auth login" in text
        or "gh_token" in text
        or "github_token" in text
        or "authentication token" in text
        or "authentication required" in text
        or "authentication failed" in text
        or "could not read username" in text
        or "permission denied (publickey)" in text
        # GitHub deliberately returns 404-style repository errors when a
        # credential cannot see a private repository.  At this point the repo
        # has already been validated against the connected-repository catalog.
        or "repository not found" in text
        or "http 401" in text
        or "http 403" in text
    )


def _extract_github_pr_url(value: str) -> str | None:
    match = _PR_URL_RE.search(value)
    return match.group(0) if match else None


def _pr_title(*, step: ExecutorStep, ctx: ExecutorContext) -> str:
    value = step.params.get("title")
    if isinstance(value, str) and value.strip():
        return value.strip()[:256]
    title = ctx.ticket.get("title")
    if isinstance(title, str) and title.strip():
        return title.strip()[:256]
    ticket_id = ctx.ticket.get("id")
    return f"SICKR ticket {ticket_id}" if isinstance(ticket_id, str) and ticket_id.strip() else "SICKR ticket"


def _pr_body(*, step: ExecutorStep, ctx: ExecutorContext) -> str:
    value = step.params.get("body")
    if isinstance(value, str) and value.strip():
        return value.strip()
    ticket_id = ctx.ticket.get("id")
    if isinstance(ticket_id, str) and ticket_id.strip():
        return f"Automated pull request for SICKR ticket {ticket_id.strip()}."
    return "Automated pull request for a SICKR ticket."


def _check_is_green(check: Any) -> bool:
    if not isinstance(check, dict):
        return False
    conclusion = str(check.get("conclusion") or "").upper()
    status = str(check.get("status") or check.get("state") or "").upper()
    if conclusion:
        return conclusion in {"SUCCESS", "SKIPPED", "NEUTRAL"}
    return status in {"COMPLETED", "SUCCESS"}


def _check_is_pending(check: Any) -> bool:
    if not isinstance(check, dict):
        return False
    conclusion = str(check.get("conclusion") or "").upper()
    status = str(check.get("status") or check.get("state") or "").upper()
    if conclusion:
        return False
    return status in {"", "PENDING", "QUEUED", "REQUESTED", "WAITING", "IN_PROGRESS", "EXPECTED"}


def _bounded_positive_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        parsed = int(value)
    elif isinstance(value, str) and value.strip().isdigit():
        parsed = int(value.strip())
    else:
        return default
    return min(max(parsed, minimum), maximum)


def _run_script_step_executor(
    *,
    step: ExecutorStep,
    ctx: ExecutorContext,
    actor_result: dict[str, Any] | None,
) -> ExecutorStepResult:
    try:
      resolved = SourceResolver(cache_root=ctx.workspace_root / ".sickr" / "sources").resolve(step.source or {})
    except WorkflowCodeSourceError as err:
        return ExecutorStepResult(step.executor_id, "error", f"executor source error: {err}")

    runtime_dir = ctx.workspace_root / ".sickr" / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    timeout = step.timeout_seconds or 600
    payload = _executor_input_payload(step=step, ctx=ctx, actor_result=actor_result, source_provenance=resolved.provenance)
    with tempfile.TemporaryDirectory(prefix="executor-step-", dir=str(runtime_dir)) as tmp:
        tmp_path = Path(tmp)
        input_path = tmp_path / "input.json"
        output_path = tmp_path / "output.json"
        input_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        try:
            proc = run_process(
                [sys.executable, str(resolved.script_path), "--input", str(input_path), "--output", str(output_path)],
                cwd=resolved.source_root,
                env=_controlled_env(ctx.env),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return ExecutorStepResult(step.executor_id, "error", f"executor timed out after {timeout}s", {"source_provenance": resolved.provenance})

        stdout = _truncate(proc.stdout or "", 64 * 1024)
        stderr = _truncate(proc.stderr or "", 64 * 1024)
        base_evidence = {"source_provenance": resolved.provenance, "stdout": stdout, "stderr": stderr}
        if proc.returncode != 0:
            return ExecutorStepResult(step.executor_id, "error", f"executor exited with code {proc.returncode}", base_evidence)
        if not output_path.exists():
            return ExecutorStepResult(step.executor_id, "error", "executor did not write output JSON", base_evidence)
        raw = output_path.read_text(encoding="utf-8")
        if len(raw.encode("utf-8")) > 1024 * 1024:
            return ExecutorStepResult(step.executor_id, "error", "executor output JSON exceeds 1MB", base_evidence)
        try:
            output = json.loads(raw)
        except json.JSONDecodeError:
            return ExecutorStepResult(step.executor_id, "error", "executor output JSON is invalid", base_evidence)
        return _executor_result_from_protocol_output(step=step, output=output, base_evidence={"executor_transport": base_evidence}, nest_evidence=False)


def _executor_input_payload(
    *, step: ExecutorStep, ctx: ExecutorContext, actor_result: dict[str, Any] | None,
    source_provenance: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema_version": "sickr.executor.input.v1",
        "executor_id": step.executor_id,
        "executor_contract_version": step.executor_contract_version,
        "ticket": ctx.ticket,
        "workspace": {"path": str(ctx.workspace_root)},
        "params": step.params,
        "actor_result": actor_result,
        "prior_step_results": list(ctx.prior_step_results),
        "source_provenance": source_provenance,
    }


def _executor_result_from_protocol_output(
    *, step: ExecutorStep, output: Any, base_evidence: dict[str, Any], nest_evidence: bool,
) -> ExecutorStepResult:
    if not isinstance(output, dict) or output.get("schema_version") != "sickr.executor.output.v1":
        return ExecutorStepResult(step.executor_id, "error", "executor output schema_version must be sickr.executor.output.v1", base_evidence)
    status = output.get("status")
    if status not in {"passed", "failed", "error", "skipped"}:
        return ExecutorStepResult(step.executor_id, "error", "executor status must be passed, failed, error, or skipped", base_evidence)
    evidence = output.get("evidence") if isinstance(output.get("evidence"), dict) else {}
    if "executor_transport" in evidence:
        return ExecutorStepResult(step.executor_id, "error", "executor evidence uses reserved key executor_transport", base_evidence)
    projected = {**base_evidence, "executor": evidence} if nest_evidence else {**evidence, **base_evidence}
    return ExecutorStepResult(step.executor_id, status, str(output.get("message") or "")[:4096], projected)  # type: ignore[arg-type]


def _run_builtin_step_executor(
    *, executor: StepExecutor, step: ExecutorStep, ctx: ExecutorContext, actor_result: dict[str, Any] | None,
) -> ExecutorStepResult:
    """Run bundled code through the same versioned envelope as repo code."""
    payload = _executor_input_payload(step=step, ctx=ctx, actor_result=actor_result, source_provenance={"source": "sickr_default"})
    try:
        json.loads(json.dumps(payload, sort_keys=True))
    except (TypeError, ValueError) as err:
        return ExecutorStepResult(step.executor_id, "error", f"executor input is not JSON-compatible: {err}")
    result = executor(step=step, ctx=ctx, actor_result=actor_result)
    output = json.loads(json.dumps({
        "schema_version": "sickr.executor.output.v1",
        "status": result.status,
        "message": result.message,
        "evidence": result.evidence,
    }, sort_keys=True))
    return _executor_result_from_protocol_output(step=step, output=output, base_evidence={}, nest_evidence=False)


def _bounded_timeout(value: Any) -> int | None:
    if isinstance(value, (int, float)):
        timeout = int(value)
        if 1 <= timeout <= 3600:
            return timeout
    return None


def _sweep_generated_artifacts(
    *,
    step: ExecutorStep,
    ctx: ExecutorContext,
    workdir: Path,
    status_stdout: str,
    timeout: int,
) -> tuple[list[str], str]:
    """Remove untracked paths matching the declared allowlist; report the rest.

    Only ``??`` (untracked) entries are candidates, and only when the step
    declares ``generated_artifact_globs``. Tracked modifications are never
    swept — ``M src/api/index.ts`` and ``?? run_tests.cjs`` want opposite
    responses, which is the whole reason this classification exists. Returns
    (swept paths, remaining porcelain output).
    """
    raw_globs = step.params.get("generated_artifact_globs")
    if not isinstance(raw_globs, list):
        return [], status_stdout
    globs = [str(g).strip() for g in raw_globs if isinstance(g, str) and str(g).strip()]
    if not globs:
        return [], status_stdout

    to_sweep: list[str] = []
    for line in status_stdout.splitlines():
        if not line.startswith("?? "):
            continue
        path = line[3:].strip().strip('"')
        normalized = path.replace("\\", "/").rstrip("/")
        if any(fnmatch.fnmatchcase(normalized, glob) for glob in globs):
            to_sweep.append(path)
    if not to_sweep:
        return [], status_stdout

    clean = _git_step(
        ["clean", "-fdq", "--", *to_sweep], cwd=workdir, env=ctx.env, timeout=timeout
    )
    if clean.returncode != 0:
        # Sweeping is an assist, not a gate of its own: if it fails, the
        # original dirt report stands and the step fails on it as before.
        return [], status_stdout
    after = _git_step(["status", "--porcelain"], cwd=workdir, env=ctx.env, timeout=timeout)
    if after.returncode != 0:
        return to_sweep, status_stdout
    return to_sweep, after.stdout


def _dirty_worktree_summary(status_stdout: str, *, limit: int = 6) -> str:
    """Render `git status --porcelain` into a one-line cause for the operator.

    "requires a clean worktree" alone forces whoever is triaging to open the
    attempt evidence to learn which paths are dirty. The usual cause is build
    output from a preceding ci.run_* step that the repository does not ignore,
    and naming the paths makes that obvious at a glance.
    """
    entries = [line.strip() for line in status_stdout.splitlines() if line.strip()]
    if not entries:
        return ""
    untracked = sum(1 for e in entries if e.startswith("??"))
    shown = ", ".join(entries[:limit])
    suffix = f" (+{len(entries) - limit} more)" if len(entries) > limit else ""
    hint = ""
    if untracked == len(entries):
        hint = " — all untracked; if these are build artifacts, add them to .gitignore"
    return f"{shown}{suffix}{hint}"


def _controlled_env(env: dict[str, str]) -> dict[str, str]:
    allowed: dict[str, str] = {}
    for key in (
        "PATH",
        "SystemRoot",
        "WINDIR",
        "TEMP",
        "TMP",
        "HOME",
        "USERPROFILE",
    ):
        # Ambient fallback is fine for these: they are host wiring, not identity.
        value = env.get(key) or os.environ.get(key)
        if value:
            allowed[key] = value
    # Shell-based CI commands such as ``python -m pytest`` must resolve to the
    # interpreter that launched the orchestrator, not the Windows Store Python
    # install-manager alias. The alias may install a new interpreter into the
    # repository and then fail because that interpreter has no test packages.
    runtime_bin = str(Path(sys.executable).resolve().parent) if sys.executable else ""
    if runtime_bin:
        existing_path = allowed.get("PATH", "")
        path_parts = [part for part in existing_path.split(os.pathsep) if part]
        compare = runtime_bin.lower() if os.name == "nt" else runtime_bin
        path_parts = [
            part for part in path_parts
            if (part.lower() if os.name == "nt" else part) != compare
        ]
        allowed["PATH"] = os.pathsep.join([runtime_bin, *path_parts])
    for key in (
        "GIT_TERMINAL_PROMPT",
        "GIT_CONFIG_COUNT",
        "GIT_CONFIG_KEY_0",
        "GIT_CONFIG_VALUE_0",
        # GitHub identity, forwarded ONLY when the caller put it in `env`.
        # That is the lease-scoped credential github_token_env injects; it is
        # never read from os.environ, so an operator's shell login cannot
        # stand in for a missing lease. Omitting it here stripped the brokered
        # token back out before `gh` ran, which surfaced as
        # "To get started with GitHub CLI, please run: gh auth login".
        "GH_TOKEN",
        "GITHUB_TOKEN",
        "GITHUB_ENTERPRISE_TOKEN",
    ):
        value = env.get(key)
        if value:
            allowed[key] = value
    return allowed


def _git_step(args: list[str], *, cwd: Path, env: dict[str, str], timeout: int) -> GitStepResult:
    try:
        proc = run_process(
            ["git", *args],
            cwd=cwd,
            env=_controlled_env(env),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as err:
        return GitStepResult(
            returncode=124,
            stdout=err.stdout if isinstance(err.stdout, str) else "",
            stderr=err.stderr if isinstance(err.stderr, str) else f"git command timed out after {timeout}s",
        )
    return GitStepResult(proc.returncode, proc.stdout or "", proc.stderr or "")


def _git_remote_ref_missing(result: GitStepResult) -> bool:
    diagnostic = f"{result.stderr}\n{result.stdout}".casefold()
    return any(
        marker in diagnostic
        for marker in (
            "couldn't find remote ref",
            "could not find remote ref",
            "remote ref does not exist",
        )
    )


def _branch_param(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    branch = value.strip()
    if not branch or branch.startswith("-") or ".." in branch or branch.endswith("/") or branch.startswith("/"):
        return None
    if any(ch in branch for ch in ("\0", "\\", "~", "^", ":", "?", "*", "[")):
        return None
    return branch


def _truncate(value: str, limit: int) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= limit:
        return value
    return encoded[:limit].decode("utf-8", errors="replace")


def _nonempty_param(params: dict[str, Any], key: str) -> str | None:
    value = params.get(key)
    return value.strip() if isinstance(value, str) and value.strip() else None


def _ci_commands(params: dict[str, Any]) -> list[str]:
    command = params.get("command")
    if isinstance(command, str) and command.strip():
        return [command.strip()]
    commands = params.get("commands")
    if isinstance(commands, list):
        return [value.strip() for value in commands if isinstance(value, str) and value.strip()]
    return []


def _ci_working_directory(ctx: ExecutorContext, value: Any) -> Path | None:
    root = ctx.workspace_root.resolve()
    if not isinstance(value, str) or not value.strip():
        return _default_ci_working_directory(root=root, ticket=ctx.ticket)
    candidate = (root / value.strip()).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate


def _default_ci_working_directory(*, root: Path, ticket: dict[str, Any]) -> Path:
    repo_name = _ticket_repo_dir_name(ticket)
    if repo_name:
        if root.name == repo_name and root.exists() and root.is_dir():
            return root
        for candidate in _candidate_repo_workdirs(root=root, ticket=ticket, repo_name=repo_name):
            if candidate.exists() and candidate.is_dir():
                return candidate
    return root


def _candidate_repo_workdirs(*, root: Path, ticket: dict[str, Any], repo_name: str) -> list[Path]:
    candidates = [(root / repo_name).resolve()]
    ticket_id = ticket.get("id")
    if isinstance(ticket_id, str) and ticket_id.strip():
        candidates.append((root / ".sickr-workspaces" / ticket_id.strip() / repo_name).resolve())

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            candidate.relative_to(root)
        except ValueError:
            continue
        key = str(candidate).casefold()
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def _ticket_repo_dir_name(ticket: dict[str, Any]) -> str | None:
    for key in ("repo", "repository", "repo_slug", "repository_slug", "repo_url", "repository_url"):
        value = ticket.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        candidate = value.strip().rstrip("/")
        if candidate.endswith(".git"):
            candidate = candidate[:-4]
        if "/" in candidate:
            candidate = candidate.rsplit("/", 1)[-1]
        if re.fullmatch(r"[A-Za-z0-9._-]+", candidate):
            return candidate
    return None


def _nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _resolve_dotted_path(root: Any, path: str) -> tuple[bool, Any]:
    current = root
    for part in path.split("."):
        if isinstance(current, dict):
            if part not in current:
                return False, None
            current = current[part]
        elif isinstance(current, list):
            # Numeric index into a list, e.g. "pr_links.0" -> pr_links[0].
            # Without this, evidence.has_any paths that index a list (the merge
            # preflight's `pr_links.0` PR-reference check) never resolve even
            # when the element exists.
            if not part.isdigit():
                return False, None
            idx = int(part)
            if idx >= len(current):
                return False, None
            current = current[idx]
        else:
            return False, None
    return True, current


# Statuses that count a fix ticket as complete (mirrors the workflow-service
# dependency gate DEPENDENCY_COMPLETE_STATUSES). Merged-but-unvalidated
# (COMPLETED/VALIDATE) does NOT unblock — the fix must fully land.
_REMEDIATION_COMPLETE_STATUSES = frozenset({"DONE", "RESOLVED", "CLOSED"})
_REMEDIATION_TERMINAL_FAILURE_STATUSES = frozenset({"FAILED", "CANCELLED"})


def _remediation_spawn_tickets(
    *, step: ExecutorStep, ctx: ExecutorContext, actor_result: dict[str, Any] | None
) -> ExecutorStepResult:
    """Spawn one fix ticket per repo from a failed evaluation's findings.

    A SIDE-EFFECT step: it never fails the state (operators configure
    on_failure=continue, but the executor also always returns ``passed`` so
    it can sit FIRST in a postflight list without blocking the deciding
    step). It acts only on a failure outcome; on success or when there is
    nothing actionable it no-ops, recording the reason in evidence.

    On failure it groups blocking findings (each carrying a ``repo``) by
    repository, creates one fix ticket (R) per repo — DRAFT by default, or
    READY via ``params.spawn_status`` (which promotes through the backend
    Draft->Ready gate so the repo's default workflow binds) — and PATCHes
    the validation ticket (V) ONCE with the new dependencies (V depends_on
    R) + links + the ``remediation`` stamp. The workflow-service park hook
    reads the stamp from the submitted failing evidence and parks V WAITING
    at its current verification node.

    When a Remedy ticket (R) completes, Verification (V) wakes and re-runs:
    workflow-service's wakeRemediationDependents flips V WAITING->READY at
    the SAME node once all its R are complete, and the next lease re-runs
    the verification state from its preflight (git.sync re-pulls the merged
    fixes) — not a whole-workflow restart, not a mid-state resume. If an R
    instead ends FAILED/CANCELLED, the deadman escalates V (it can never
    wake on its own).
    """
    remediation: dict[str, Any] = {}

    outcome = _remediation_main_outcome(actor_result)
    if outcome != "failure":
        remediation["skipped"] = f"outcome_not_failure:{outcome}"
        return _remediation_result(step, remediation)

    ticket = ctx.ticket if isinstance(ctx.ticket, dict) else {}
    ticket_id = str(ticket.get("id") or "").strip()
    if not ticket_id:
        remediation["skipped"] = "missing_ticket_id"
        return _remediation_result(step, remediation)
    if ctx.workflow_client is None or not hasattr(ctx.workflow_client, "create_ticket"):
        remediation["skipped"] = "no_workflow_client"
        return _remediation_result(step, remediation)

    params = step.params if isinstance(step.params, dict) else {}
    # max_cycles bounds the fail -> fix -> re-verify loop. Each round where V
    # wakes (its R all completed) and STILL fails is one cycle; at the limit
    # (default 2) the executor stops spawning and lets the state's normal
    # failure route escalate to a human — the safety valve against an
    # infinite orbit on a defect the fleet can't actually resolve. This is
    # distinct from the state's own failure_policy retry (retry_same_state),
    # which re-runs the SAME state immediately with no code change (for
    # transient/flaky failures); a real defect needs a fix authored between
    # attempts, which is what a cycle buys.
    try:
        max_cycles = int(params.get("max_cycles", 2))
    except (TypeError, ValueError):
        max_cycles = 2
    # Final state of spawned fix tickets: DRAFT (default) or READY. A READY
    # spawn is created DRAFT then promoted through the backend's Draft->Ready
    # gate, which resolves the workflow from the repo's defaults and enforces
    # the binding — so an unconfigured repo safely leaves the ticket DRAFT.
    spawn_status = str(params.get("spawn_status") or "DRAFT").upper()
    if spawn_status not in ("DRAFT", "READY"):
        spawn_status = "DRAFT"

    # Re-fetch the ticket so the prior remediation stamp + depends_on reflect
    # any earlier round (ctx.ticket is a lease-time snapshot).
    current = ticket
    if hasattr(ctx.workflow_client, "get_ticket"):
        try:
            fetched = ctx.workflow_client.get_ticket(ticket_id)
            if isinstance(fetched, dict):
                current = fetched
        except Exception:  # noqa: BLE001 - fall back to the snapshot on any read fault
            current = ticket

    prior = current.get("evidence")
    prior_remediation = prior.get("remediation") if isinstance(prior, dict) and isinstance(prior.get("remediation"), dict) else {}
    prior_spawned = prior_remediation.get("spawned") if isinstance(prior_remediation.get("spawned"), dict) else {}
    try:
        cycle = int(prior_remediation.get("cycle", 0))
    except (TypeError, ValueError):
        cycle = 0

    # Cycle bound: a re-run that fails while a prior spawn map exists is a
    # wake-and-fail round — the fixes landed and verification still fails.
    if prior_spawned:
        cycle += 1
        if cycle >= max_cycles:
            remediation.update({"cycles_exhausted": True, "cycle": cycle, "spawned": dict(prior_spawned)})
            _remediation_patch(ctx, ticket_id, current, remediation)
            return _remediation_result(step, remediation)

    findings = _remediation_findings(actor_result)
    by_repo: dict[str, list[dict[str, Any]]] = {}
    covered_finding_ids = set(prior_spawned.keys())
    for finding in findings:
        severity = str(finding.get("severity") or "blocking").lower()
        if severity != "blocking":
            continue
        repo = str(finding.get("repo") or "").strip()
        finding_id = str(finding.get("id") or "").strip()
        if not repo:
            continue
        if finding_id and finding_id in covered_finding_ids:
            continue  # dedup: already spawned in a prior round
        by_repo.setdefault(repo, []).append(finding)

    # Fix tickets spawn as DRAFT with just the repo — no per-state workflow
    # routing. The repo's configured defaults (default workflow / branch /
    # priority) resolve when a human moves the ticket DRAFT->READY, gated so
    # it cannot dispatch without a proper binding.
    spawned: dict[str, str] = dict(prior_spawned)
    new_ticket_ids: list[str] = []
    create_failures: dict[str, str] = {}
    not_promoted: list[str] = []
    for repo, repo_findings in by_repo.items():
        payload = _remediation_create_payload(current, repo, repo_findings)
        try:
            created = ctx.workflow_client.create_ticket(payload)
        except Exception as err:  # noqa: BLE001 - a create fault is per-repo evidence, not a state failure
            create_failures[repo] = repr(err)[:400]
            continue
        new_id = str(created.get("id") or "").strip() if isinstance(created, dict) else ""
        if not new_id:
            create_failures[repo] = "create returned no ticket id"
            continue
        new_ticket_ids.append(new_id)
        for finding in repo_findings:
            fid = str(finding.get("id") or "").strip() or f"{repo}#{len(spawned)}"
            spawned[fid] = new_id
        # Optional promote to READY through the Draft->Ready gate. A repo with
        # no configured default workflow can't bind, so the promote fails and
        # the ticket stays DRAFT for a human — recorded, never fatal.
        if spawn_status == "READY" and hasattr(ctx.workflow_client, "update_ticket"):
            try:
                ctx.workflow_client.update_ticket(new_id, {"status": "READY"})
            except Exception:  # noqa: BLE001 - unbindable repo leaves the fix ticket DRAFT
                not_promoted.append(new_id)

    remediation.update({"spawned": spawned, "cycle": cycle, "parked_at": current.get("workflow_node_id")})
    if create_failures:
        remediation["create_failures"] = create_failures
    if not_promoted:
        remediation["not_promoted"] = not_promoted

    if new_ticket_ids or (spawned and not prior_spawned):
        _remediation_patch(ctx, ticket_id, current, remediation, new_ticket_ids)
    return _remediation_result(step, remediation)


def _remediation_result(step: ExecutorStep, remediation: dict[str, Any]) -> ExecutorStepResult:
    # Always passed: this step must never alter the state's own outcome.
    return ExecutorStepResult(step.executor_id, "passed", "remediation evaluated", {"remediation": remediation})


def _remediation_main_outcome(actor_result: dict[str, Any] | None) -> str:
    if not isinstance(actor_result, dict):
        return "unknown"
    evidence = actor_result.get("evidence") if isinstance(actor_result.get("evidence"), dict) else {}
    for key in ("node_outcome", "validation_outcome"):
        value = evidence.get(key)
        if isinstance(value, str) and value:
            return value
    # Fall back to the actor status: a non-completed actor is a failure.
    return "failure" if actor_result.get("status") not in {"completed", None} else "unknown"


def _remediation_findings(actor_result: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(actor_result, dict):
        return []
    evidence = actor_result.get("evidence") if isinstance(actor_result.get("evidence"), dict) else {}
    raw = evidence.get("findings")
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def _remediation_create_payload(
    validation_ticket: dict[str, Any], repo: str, findings: list[dict[str, Any]]
) -> dict[str, Any]:
    repo_name = repo.split("/")[-1] or repo
    title = f"Fix: {validation_ticket.get('title') or 'validation'} — {repo_name}"
    lines: list[str] = []
    criteria: list[str] = []
    for finding in findings:
        fid = str(finding.get("id") or "").strip()
        heading = f"## {fid}: {finding.get('title') or ''}".strip()
        lines.append(heading)
        if finding.get("expected"):
            lines.append(f"Expected: {finding.get('expected')}")
        if finding.get("observed"):
            lines.append(f"Observed: {finding.get('observed')}")
        lines.append("")
        raw_criteria = finding.get("acceptance_criteria")
        if isinstance(raw_criteria, list):
            criteria.extend(str(c) for c in raw_criteria if str(c).strip())
    # DRAFT + a legacy placeholder type: the repo's default workflow binds
    # when a human promotes it to READY (the backend gate resolves + enforces).
    payload: dict[str, Any] = {
        "title": title,
        "type": "chore",
        "repo": repo,
        "status": "DRAFT",
        "description": "\n".join(lines).strip(),
        "acceptance_criteria": criteria,
    }
    if validation_ticket.get("parent_id"):
        payload["parent_id"] = validation_ticket.get("parent_id")
    if validation_ticket.get("priority"):
        payload["priority"] = validation_ticket.get("priority")
    return payload


def _remediation_patch(
    ctx: ExecutorContext,
    ticket_id: str,
    current: dict[str, Any],
    remediation: dict[str, Any],
    new_ticket_ids: list[str] | None = None,
) -> None:
    """One PATCH carrying deps + links + the remediation stamp (the backend
    evidence merge is shallow top-level, so everything travels together)."""
    if not hasattr(ctx.workflow_client, "update_ticket"):
        return
    payload: dict[str, Any] = {"evidence": {"remediation": remediation}}
    if new_ticket_ids:
        existing_deps = current.get("depends_on_ticket_ids")
        existing_deps = list(existing_deps) if isinstance(existing_deps, list) else []
        existing_links = current.get("linked_tickets")
        existing_links = list(existing_links) if isinstance(existing_links, list) else []
        payload["depends_on_ticket_ids"] = _dedupe_preserve(existing_deps + new_ticket_ids)
        payload["linked_tickets"] = _dedupe_preserve(existing_links + new_ticket_ids)
    try:
        ctx.workflow_client.update_ticket(ticket_id, payload)
    except Exception as err:  # noqa: BLE001 - a PATCH fault is recorded, not raised
        remediation["patch_error"] = repr(err)[:400]


def _dedupe_preserve(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        text = str(item)
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


DEFAULT_STEP_EXECUTORS: dict[str, StepExecutor] = {
    "ticket.has_description": _ticket_has_description,
    "ticket.has_acceptance_criteria": _ticket_has_acceptance_criteria,
    "ticket.has_workflow": _ticket_has_workflow,
    "ticket.has_repo": _ticket_has_repo,
    "ticket.has_branch": _ticket_has_branch,
    "ticket.has_priority": _ticket_has_priority,
    "workspace.exists": _workspace_exists,
    "actor.completed": _actor_completed,
    "channel.require": _channel_require,
    "evidence.has_any": _evidence_has_any,
    "git.ensure_ticket_branch": _git_ensure_ticket_branch,
    "git.sync_with_base": _git_sync_with_base,
    "git.commit_state_changes": _git_commit_state_changes,
    "git.ensure_published": _git_ensure_published,
    "ci.run_tests": _ci_run_tests,
    "ci.run_build": _ci_run_build,
    "ci.run_typecheck": _ci_run_typecheck,
    "ci.run_command": _ci_run_custom_command,
    "package.install_dependencies": _package_install_dependencies,
    "workspace.clone_repositories": _workspace_clone_repositories,
    "validation.run_baseline": _validation_run_baseline,
    "workspace.capture_generated_ignores": _workspace_capture_generated_ignores,
    "integration.dispatch": _integration_dispatch,
    "github.publish_and_ensure_pr": _github_publish_and_ensure_pr,
    "github.prepare_pr_workspace": _github_prepare_pr_workspace,
    "github.sync_review_decision": _github_sync_review_decision,
    "github.ensure_deployed": _github_ensure_deployed,
    "pr.extract_context": _pr_extract_context,
    "pr.approved": _pr_approved,
    "pr.checks_green": _pr_checks_green,
    "pr.mergeable": _pr_mergeable,
    "remediation.spawn_tickets": _remediation_spawn_tickets,
}

_STEP_EXECUTOR_MANIFESTS = {
    executor_id: descriptor
    for executor_id, descriptor in BUILTIN_EXECUTOR_MANIFESTS.items()
    if any(phase in descriptor.supported_phases for phase in ("preflight", "postflight"))
}
assert_registration_conformance(
    descriptors=_STEP_EXECUTOR_MANIFESTS,
    registrations=DEFAULT_STEP_EXECUTORS,
    registration_name="DEFAULT_STEP_EXECUTORS",
)
