"""Built-in deterministic operations for ExecutionContract main steps.

workflow-service chooses the operation by stable ``operation_id`` and
owns the evidence-to-transition policy. This module only runs local
platform primitives and returns raw evidence.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

from labudi_orchestrator.agent_workflow_client import AgentWorkflowClient
from labudi_orchestrator.execution_contract_runner import (
    DeterministicMainExecutor,
    MainExecutorResult,
)
from sickr_workflow_executors.executor_manifest import (
    BUILTIN_EXECUTOR_MANIFESTS,
    assert_registration_conformance,
)
from labudi_orchestrator.github_credential_environment import (
    GitHubCredentialEnvironmentError,
    resolve_github_credential_environment,
)
from labudi_orchestrator.obligation_executors import ExecutorContext
from labudi_orchestrator.proc import run_process
from labudi_orchestrator.ship_pipeline import ship
from labudi_orchestrator.ticket_planning import plan_to_ticket_specs
from labudi_orchestrator.workflow_code_source import SourceResolver, WorkflowCodeSourceError
from sickr_workflow_executors.workflow_executor_steps import run_executor_steps


def build_deterministic_main_executor(
    *,
    client: AgentWorkflowClient,
    agent_id: str,
    workspace_root: Path,
    env: Mapping[str, str],
    log_fn,
) -> DeterministicMainExecutor:
    """Create the built-in deterministic executor registry."""
    operations = {
        "runtime.noop": _runtime_noop,
        "ship_pipeline_v1": lambda *, spec, ctx: _ship_pipeline_v1(
            spec=spec,
            ctx=ctx,
            client=client,
            agent_id=agent_id,
            workspace_root=workspace_root,
            env=env,
            log_fn=log_fn,
        ),
        "skip_validation_v1": _skip_validation_v1,
        "create_tickets_v1": lambda *, spec, ctx: _create_tickets_v1(
            spec=spec,
            ctx=ctx,
            client=client,
        ),
        "script_worker_v1": _script_worker_v1,
        "executor_steps_v1": _executor_steps_v1,
        "sources.collect_v1": lambda *, spec, ctx: _sources_collect_v1(spec=spec, ctx=ctx, client=client),
        "records.normalize_v1": _records_normalize_v1,
        "records.rank_v1": _records_rank_v1,
        "records.select_v1": _records_select_v1,
        "signals.ingest_ops_v1": lambda *, spec, ctx: _signals_ingest_ops_v1(spec=spec, ctx=ctx, client=client),
    }
    main_descriptors = {
        executor_id: descriptor
        for executor_id, descriptor in BUILTIN_EXECUTOR_MANIFESTS.items()
        if "main" in descriptor.supported_phases
    }
    assert_registration_conformance(
        descriptors=main_descriptors,
        registrations=operations,
        registration_name="deterministic main operations",
    )
    return DeterministicMainExecutor(operations, external_operation=_external_executor_main)


def _external_executor_main(*, spec: dict[str, Any], ctx: ExecutorContext) -> MainExecutorResult:
    outcome, results = run_executor_steps(config_json={"steps": [{
        "executor_id": spec.get("operation_id"),
        "executor_contract_version": spec.get("executor_contract_version"),
        "source": spec.get("executor_source"),
        "params": spec.get("params") if isinstance(spec.get("params"), dict) else {},
        "timeout_seconds": spec.get("timeout_seconds"),
        "required": True,
        "on_failure": "fail",
    }]}, ctx=ctx)
    result = results[0] if results else None
    if outcome in {"passed", "skipped"} and result is not None:
        return MainExecutorResult(status="completed", evidence=result.evidence)
    return MainExecutorResult(status="failed", evidence=result.evidence if result is not None else {}, failure_reason=result.message if result is not None else "external executor produced no result")


def _runtime_noop(*, spec: dict[str, Any], ctx: ExecutorContext) -> MainExecutorResult:
    """Complete a deterministic state whose work lives in flight executors."""
    del spec, ctx
    return MainExecutorResult(status="completed", evidence={})


# Evidence keys, most-canonical first, under which the planning state's output
# (the plan) is expected to arrive on the ticket. The worker reads its input
# from the prior state's output carried on the ticket context — it does NOT
# depend on workflow-service or the backend injecting the plan into the
# contract params. That keeps the routing/execution layers out of the business
# of threading state I/O (see docs context-model plan).
_PLAN_EVIDENCE_KEYS = ("ticket_creation_plan", "plan", "feature_plan", "plan_output")


def _looks_like_plan(value: Any) -> bool:
    """A plan is any object carrying a list of nodes (edges optional)."""
    return isinstance(value, dict) and isinstance(value.get("nodes"), list)


def _resolve_plan(spec: dict[str, Any], ctx: ExecutorContext) -> dict[str, Any] | None:
    """Find the planning-plan for the Ticket Creation worker state.

    Canonical source: the planning state's output, carried forward on the
    ticket's accumulated evidence/context (``ctx.ticket.evidence``). The worker
    state naturally receives the previous state's output — nothing upstream has
    to stuff the plan into the contract.

    ``spec.params.plan`` is honored as an optional explicit override (e.g. a
    one-off invocation), but is never required. Returning None lets the caller
    fail loudly rather than create nothing silently.
    """
    params = spec.get("params")
    if isinstance(params, dict) and _looks_like_plan(params.get("plan")):
        return params["plan"]

    evidence = ctx.ticket.get("evidence")
    if isinstance(evidence, dict):
        for key in _PLAN_EVIDENCE_KEYS:
            if _looks_like_plan(evidence.get(key)):
                return evidence[key]
        # Resilient fallback: pick up a plan-shaped output regardless of the key
        # the planning state happened to use, so we don't hardcode a contract.
        for value in evidence.values():
            if _looks_like_plan(value):
                return value
    return None


def _create_tickets_v1(
    *,
    spec: dict[str, Any],
    ctx: ExecutorContext,
    client: AgentWorkflowClient,
) -> MainExecutorResult:
    """Materialize child tickets from a planning ticket's plan.

    Deterministic worker step: maps each plan node to a child ticket (type
    chosen by ``select_ticket_type``), creates them all, then wires dependency
    edges in a second update pass (so keys resolve to real ticket ids). Raw
    create/update is delegated to the backend client; this step returns the
    created (key, id, type) set as evidence for workflow-service to route on.

    Not idempotent across retries — a re-run would create duplicates. The
    workflow-service contract should only schedule this once per planning
    ticket; the created set is returned so a future guard can dedupe by key.
    """
    plan = _resolve_plan(spec, ctx)
    parent_ticket_id = str(ctx.ticket.get("id") or "") or None
    if not isinstance(plan, dict):
        if not parent_ticket_id:
            return MainExecutorResult(
                status="failed",
                failure_reason="create_tickets_v1: no inline plan and no ticket id to materialize a feature plan for",
            )
        return _materialize_feature_plan(client=client, ticket_id=parent_ticket_id)

    try:
        specs = plan_to_ticket_specs(plan, parent_ticket_id=parent_ticket_id)
    except ValueError as err:
        return MainExecutorResult(
            status="failed",
            failure_reason=f"create_tickets_v1: invalid plan: {err}",
        )

    key_to_id: dict[str, str] = {}
    created: list[dict[str, Any]] = []
    parent_run_kind = ctx.ticket.get("run_kind")
    parent_run_id = ctx.ticket.get("run_id")
    for planned in specs:
        payload: dict[str, Any] = {
            "title": planned.title,
            "description": planned.description,
            "type": planned.type,
            "priority": planned.priority,
            "status": planned.status,
            "acceptance_criteria": planned.acceptance_criteria,
            "evidence": {"ticket_creation": planned.evidence},
        }
        if parent_ticket_id:
            payload["parent_id"] = parent_ticket_id
        if isinstance(parent_run_kind, str) and parent_run_kind:
            payload["run_kind"] = parent_run_kind
        if isinstance(parent_run_id, str) and parent_run_id:
            payload["run_id"] = parent_run_id
        body = client.create_ticket(payload)
        raw_id = body.get("id") if isinstance(body, dict) else None
        ticket_id = str(raw_id) if raw_id else ""
        if not ticket_id:
            return MainExecutorResult(
                status="failed",
                failure_reason=(
                    f"create_tickets_v1: backend returned no id for plan node {planned.key!r}; "
                    f"created {len(created)} of {len(specs)} before aborting"
                ),
            )
        key_to_id[planned.key] = ticket_id
        created.append({"key": planned.key, "id": ticket_id, "type": planned.type})

    for planned in specs:
        if not planned.depends_on_keys:
            continue
        dep_ids = [key_to_id[k] for k in planned.depends_on_keys if k in key_to_id]
        if dep_ids:
            client.update_ticket(key_to_id[planned.key], {"depends_on_ticket_ids": dep_ids})

    return MainExecutorResult(
        status="completed",
        evidence={
            "ticket_creation_outcome": "completed",
            "created_tickets": created,
            "created_count": len(created),
            "node_outcome": "success",
        },
    )


def _materialize_feature_plan(
    *,
    client: AgentWorkflowClient,
    ticket_id: str,
) -> MainExecutorResult:
    """Materialize the ticket's feature plan via the backend.

    The backend endpoint is agent-key scoped and owns lifecycle/idempotency
    checks. A rejection is reported as deterministic worker failure evidence
    instead of silently creating no tickets.
    """
    try:
        result = client.materialize_feature_plan(ticket_id)
    except Exception as err:  # noqa: BLE001 - backend rejection is evidence, not a crash
        return MainExecutorResult(
            status="failed",
            failure_reason=f"create_tickets_v1: feature-plan materialize failed: {err!r}",
        )
    if not isinstance(result, dict) or not result.get("ok"):
        return MainExecutorResult(
            status="failed",
            failure_reason=f"create_tickets_v1: no feature plan to materialize for ticket {ticket_id}",
        )
    materialized = result.get("materialized") or []
    created = [
        {"key": item.get("node_id"), "id": item.get("ticket_id"), "source": "feature_plan"}
        for item in materialized
        if isinstance(item, dict)
    ]
    return MainExecutorResult(
        status="completed",
        evidence={
            "ticket_creation_outcome": "completed",
            "created_tickets": created,
            "created_count": len(created),
            "source": "feature_plan",
            "feature_plan_id": result.get("plan_id"),
            "node_outcome": "success",
        },
    )


def _script_worker_v1(*, spec: dict[str, Any], ctx: ExecutorContext) -> MainExecutorResult:
    params = spec.get("params")
    if not isinstance(params, dict):
        return MainExecutorResult(status="failed", failure_reason="script_worker_v1 requires params")
    source = params.get("effective_code_source") or params.get("configured_code_source")
    if not isinstance(source, dict):
        return MainExecutorResult(status="failed", failure_reason="script_worker_v1 requires effective_code_source")

    try:
        resolved = SourceResolver(cache_root=ctx.workspace_root / ".sickr" / "sources").resolve(source)
    except WorkflowCodeSourceError as err:
        return MainExecutorResult(status="failed", failure_reason=f"script_worker_v1 source error: {err}")

    runtime_dir = ctx.workspace_root / ".sickr" / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    timeout = _bounded_timeout(spec.get("timeout_seconds"), default=600)
    input_payload = {
        "schema_version": "sickr.worker.input.v1",
        "ticket": ctx.ticket,
        "workspace": {"path": str(ctx.workspace_root)},
        "config": params.get("config_json") if isinstance(params.get("config_json"), dict) else {},
        "source_provenance": resolved.provenance,
    }

    with tempfile.TemporaryDirectory(prefix="script-worker-", dir=str(runtime_dir)) as tmp:
        tmp_path = Path(tmp)
        input_path = tmp_path / "input.json"
        output_path = tmp_path / "output.json"
        input_path.write_text(json.dumps(input_payload, sort_keys=True), encoding="utf-8")
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
            return MainExecutorResult(status="timed_out", failure_reason=f"script_worker_v1 timed out after {timeout}s")

        stdout = _truncate(proc.stdout or "", 64 * 1024)
        stderr = _truncate(proc.stderr or "", 64 * 1024)
        if proc.returncode != 0:
            return MainExecutorResult(
                status="failed",
                evidence={"source_provenance": resolved.provenance, "stdout": stdout, "stderr": stderr},
                failure_reason=f"script_worker_v1 exited with code {proc.returncode}",
            )
        if not output_path.exists():
            return MainExecutorResult(
                status="failed",
                evidence={"source_provenance": resolved.provenance, "stdout": stdout, "stderr": stderr},
                failure_reason="script_worker_v1 did not write output JSON",
            )
        raw = output_path.read_text(encoding="utf-8")
        if len(raw.encode("utf-8")) > 1024 * 1024:
            return MainExecutorResult(status="failed", failure_reason="script_worker_v1 output JSON exceeds 1MB")
        try:
            output = json.loads(raw)
        except json.JSONDecodeError:
            return MainExecutorResult(status="failed", failure_reason="invalid worker output JSON")
        if not isinstance(output, dict):
            return MainExecutorResult(status="failed", failure_reason="worker output must be a JSON object")
        validation_error = _validate_worker_output(output)
        if validation_error:
            return MainExecutorResult(status="failed", failure_reason=validation_error)

        evidence = output.get("evidence") if isinstance(output.get("evidence"), dict) else {}
        message = str(output.get("message") or "")[:4096]
        worker_edge = output["edge"]
        return MainExecutorResult(
            status="completed",
            evidence={
                "script_worker_edge": worker_edge,
                "script_worker_outcome": output.get("outcome"),
                "script_worker_message": message,
                "script_worker": evidence,
                "source_provenance": resolved.provenance,
                "stdout": stdout,
                "stderr": stderr,
                "node_outcome": "success" if worker_edge == "success" else "failure",
            },
        )


def _executor_steps_v1(*, spec: dict[str, Any], ctx: ExecutorContext) -> MainExecutorResult:
    params = spec.get("params")
    if not isinstance(params, dict):
        return MainExecutorResult(status="failed", failure_reason="executor_steps_v1 requires params")
    config_json = params.get("config_json") if isinstance(params.get("config_json"), dict) else {}
    # No on_progress here, deliberately. Live step progress publishes into
    # FlightView, whose `main` member carries {status, node_outcome, summary,
    # failure_reason} — not a step list. These steps run AS the main executor, so
    # the contract has nowhere to put them, and they are already invisible in the
    # SETTLED view for the same reason. Attaching a reporter would publish a shape
    # no consumer reads.
    #
    # Cheap to leave alone because nothing reaches it: as of 2026-07-29 no state
    # in production uses executor_steps_v1 as its main (deterministic mains are
    # skip_validation_v1 x5 and ship_pipeline_v1 x2). Should that change, giving
    # `main` its own step list is a contract change across four repos, not a
    # one-line callback.
    outcome, step_results = run_executor_steps(config_json=config_json, ctx=ctx, actor_result=None)
    # New per-step vocabulary: passed -> success edge; business failure -> failure
    # edge; error / system_error -> error edge.
    edge = "success" if outcome == "passed" else "skip" if outcome == "skipped" else "error" if outcome in {"error", "system_error"} else "failure"
    return MainExecutorResult(
        status="completed",
        evidence={
            "executor_steps_edge": edge,
            "executor_steps_outcome": outcome,
            "executor_steps": {
                "steps": [result.as_dict() for result in step_results],
            },
            "node_outcome": edge if edge in {"success", "skip"} else "failure",
        },
    )


def _sources_collect_v1(
    *,
    spec: dict[str, Any],
    ctx: ExecutorContext,
    client: AgentWorkflowClient,
) -> MainExecutorResult:
    params = spec.get("params")
    if not isinstance(params, dict):
        return MainExecutorResult(status="failed", failure_reason="sources.collect_v1 requires params")
    if not ctx.lease_id:
        return MainExecutorResult(status="failed", failure_reason="sources.collect_v1 requires an active lease")
    provider_id = str(params.get("provider_id") or "").strip()
    query = params.get("query")
    limit = params.get("limit", 25)
    if not provider_id or not isinstance(query, dict) or not isinstance(limit, int) or not 1 <= limit <= 100:
        return MainExecutorResult(
            status="failed",
            failure_reason="sources.collect_v1 requires provider_id, query, and limit 1..100",
        )
    try:
        result = client.collect_sources(
            ctx.lease_id,
            agent_id=ctx.agent_id,
            provider_id=provider_id,
            query=_expand_workflow_inputs(query, ctx.ticket),
            limit=limit,
            cursor=str(params["cursor"]) if params.get("cursor") else None,
        )
    except Exception as err:  # noqa: BLE001
        return MainExecutorResult(status="failed", failure_reason=f"sources.collect_v1 failed: {err!r}")
    records = result.get("records")
    if not isinstance(records, list):
        return MainExecutorResult(status="failed", failure_reason="sources.collect_v1 returned invalid records")
    return MainExecutorResult(
        status="completed",
        evidence={
            "source_records": records,
            "provider_evidence": result.get("provider_evidence") if isinstance(result.get("provider_evidence"), dict) else {},
            "next_cursor": result.get("next_cursor"),
            "source_record_count": len(records),
            "node_outcome": "success",
        },
    )


def _records_normalize_v1(*, spec: dict[str, Any], ctx: ExecutorContext) -> MainExecutorResult:
    params = spec.get("params")
    if not isinstance(params, dict):
        return MainExecutorResult(status="failed", failure_reason="records.normalize_v1 requires params")
    records = _resolve_record_list(ctx.ticket, str(params.get("input_path") or "source_records"))
    if records is None:
        return MainExecutorResult(status="failed", failure_reason="records.normalize_v1 input_path did not resolve to records")
    mapping = params.get("field_mapping")
    mapping = mapping if isinstance(mapping, dict) else {}
    required = params.get("required_fields")
    required_fields = [str(item) for item in required] if isinstance(required, list) else []
    dedupe = params.get("dedupe_keys")
    dedupe_keys = [str(item) for item in dedupe] if isinstance(dedupe, list) else ["url"]
    normalized: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            rejected.append({"index": index, "reason": "record_not_object"})
            continue
        output = dict(record)
        for target, source in mapping.items():
            if isinstance(target, str) and isinstance(source, str):
                found, value = _resolve_path(record, source)
                if found:
                    output[target] = value
        missing = [field for field in required_fields if output.get(field) in (None, "", [], {})]
        if missing:
            rejected.append({"index": index, "reason": "missing_required", "fields": missing})
            continue
        key = json.dumps([output.get(field) for field in dedupe_keys], sort_keys=True, default=str)
        if key in seen:
            rejected.append({"index": index, "reason": "duplicate"})
            continue
        seen.add(key)
        normalized.append(output)
    return MainExecutorResult(
        status="completed",
        evidence={
            "records": normalized,
            "rejected_records": rejected,
            "normalized_count": len(normalized),
            "rejected_count": len(rejected),
            "node_outcome": "success",
        },
    )


def _records_rank_v1(*, spec: dict[str, Any], ctx: ExecutorContext) -> MainExecutorResult:
    params = spec.get("params")
    if not isinstance(params, dict):
        return MainExecutorResult(status="failed", failure_reason="records.rank_v1 requires params")
    records = _resolve_record_list(ctx.ticket, str(params.get("input_path") or "records"))
    criteria = params.get("criteria")
    if records is None or not isinstance(criteria, list) or not criteria:
        return MainExecutorResult(status="failed", failure_reason="records.rank_v1 requires records and criteria")
    ranked: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        contributions: dict[str, float] = {}
        total = 0.0
        for raw in criteria:
            if not isinstance(raw, dict):
                continue
            name = str(raw.get("name") or raw.get("path") or "").strip()
            path = str(raw.get("path") or "").strip()
            weight = raw.get("weight")
            if not name or not path or not isinstance(weight, (int, float)) or weight < 0:
                continue
            found, value = _resolve_path(record, path)
            numeric = float(value) if found and isinstance(value, (int, float)) else 0.0
            normalized = max(0.0, min(1.0, numeric / 100.0 if numeric > 1 else numeric))
            contribution = round(normalized * float(weight), 6)
            contributions[name] = contribution
            total += contribution
        ranked.append({
            **record,
            "score": round(total, 6),
            "score_components": contributions,
            "_stable_index": index,
        })
    tie_breakers = params.get("tie_breakers")
    tie_paths = [str(item) for item in tie_breakers] if isinstance(tie_breakers, list) else []
    ranked.sort(key=lambda item: _rank_sort_key(item, tie_paths))
    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank
        item.pop("_stable_index", None)
    return MainExecutorResult(
        status="completed",
        evidence={"ranked_records": ranked, "ranked_count": len(ranked), "node_outcome": "success"},
    )


def _records_select_v1(*, spec: dict[str, Any], ctx: ExecutorContext) -> MainExecutorResult:
    params = spec.get("params")
    if not isinstance(params, dict):
        return MainExecutorResult(status="failed", failure_reason="records.select_v1 requires params")
    records = _resolve_record_list(ctx.ticket, str(params.get("input_path") or "ranked_records"))
    limit = params.get("limit", 5)
    minimum = params.get("minimum_score", 0)
    if records is None or not isinstance(limit, int) or not 1 <= limit <= 100 or not isinstance(minimum, (int, float)):
        return MainExecutorResult(status="failed", failure_reason="records.select_v1 params are invalid")
    eligible = [record for record in records if isinstance(record, dict) and isinstance(record.get("score"), (int, float)) and record["score"] >= minimum]
    selected = eligible[:limit]
    return MainExecutorResult(
        status="completed",
        evidence={
            "selected_records": selected,
            "selected_count": len(selected),
            "excluded_count": len(records) - len(selected),
            "empty_result": len(selected) == 0,
            "node_outcome": "success",
        },
    )


def _signals_ingest_ops_v1(
    *,
    spec: dict[str, Any],
    ctx: ExecutorContext,
    client: AgentWorkflowClient,
) -> MainExecutorResult:
    params = spec.get("params")
    if not isinstance(params, dict) or not ctx.lease_id:
        return MainExecutorResult(status="failed", failure_reason="signals.ingest_ops_v1 requires params and an active lease")
    input_path = str(params.get("input_path") or "signal_radar_run")
    found, payload = _resolve_path(ctx.ticket.get("evidence") if isinstance(ctx.ticket.get("evidence"), dict) else {}, input_path)
    if not found or not isinstance(payload, dict):
        return MainExecutorResult(status="failed", failure_reason=f"signals.ingest_ops_v1 could not resolve {input_path}")
    try:
        result = client.ingest_signal_radar(ctx.lease_id, agent_id=ctx.agent_id, payload=payload)
    except Exception as err:  # noqa: BLE001
        return MainExecutorResult(status="failed", failure_reason=f"signals.ingest_ops_v1 failed: {err!r}")
    if not result.get("id"):
        return MainExecutorResult(status="failed", failure_reason="signals.ingest_ops_v1 returned no run id")
    return MainExecutorResult(
        status="completed",
        evidence={"ops_signal_radar_ingest": result, "node_outcome": "success"},
    )


def _resolve_record_list(ticket: dict[str, Any], path: str) -> list[Any] | None:
    evidence = ticket.get("evidence")
    if not isinstance(evidence, dict):
        return None
    found, value = _resolve_path(evidence, path)
    return value if found and isinstance(value, list) else None


def _resolve_path(root: Any, path: str) -> tuple[bool, Any]:
    current = root
    for segment in [part for part in path.split(".") if part]:
        if isinstance(current, dict) and segment in current:
            current = current[segment]
        else:
            return False, None
    return True, current


def _expand_workflow_inputs(value: Any, ticket: dict[str, Any]) -> Any:
    trigger = ticket.get("trigger")
    inputs = trigger.get("workflow_inputs") if isinstance(trigger, dict) else {}
    if isinstance(value, str) and value.startswith("$workflow_inputs."):
        found, resolved = _resolve_path(inputs, value[len("$workflow_inputs."):])
        return resolved if found else None
    if isinstance(value, list):
        return [_expand_workflow_inputs(item, ticket) for item in value]
    if isinstance(value, dict):
        return {key: _expand_workflow_inputs(item, ticket) for key, item in value.items()}
    return value


def _rank_sort_key(item: dict[str, Any], tie_paths: list[str]) -> tuple[Any, ...]:
    ties: list[str] = []
    for path in tie_paths:
        found, value = _resolve_path(item, path)
        ties.append(str(value) if found else "")
    return (-float(item.get("score") or 0), *ties, int(item.get("_stable_index") or 0))


def _validate_worker_output(output: dict[str, Any]) -> str | None:
    if output.get("schema_version") != "sickr.worker.output.v1":
        return "worker output schema_version must be sickr.worker.output.v1"
    if output.get("edge") not in {"success", "failure", "error", "skip"}:
        return "worker output edge must be success, failure, error, or skip"
    outcome = output.get("outcome")
    if outcome is not None and not isinstance(outcome, str):
        return "worker output outcome must be a string"
    message = output.get("message")
    if message is not None and not isinstance(message, str):
        return "worker output message must be a string"
    evidence = output.get("evidence")
    if evidence is not None and not isinstance(evidence, dict):
        return "worker output evidence must be an object"
    return None


def _bounded_timeout(value: Any, *, default: int) -> int:
    if isinstance(value, (int, float)):
        timeout = int(value)
        if 1 <= timeout <= 3600:
            return timeout
    return default


def _controlled_env(env: Mapping[str, str]) -> dict[str, str]:
    allowed: dict[str, str] = {}
    for key in ("PATH", "SystemRoot", "WINDIR", "TEMP", "TMP", "HOME", "USERPROFILE"):
        value = env.get(key) or _os_environ_get(key)
        if value:
            allowed[key] = value
    return allowed


def _os_environ_get(key: str) -> str | None:
    import os

    return os.environ.get(key)


def _truncate(value: str, limit: int) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= limit:
        return value
    return encoded[:limit].decode("utf-8", errors="replace")


def _approval_result_matches_pr(result: dict[str, Any], pr_url: str) -> bool:
    evidence = result.get("evidence")
    if not isinstance(evidence, dict):
        return False
    candidates = [evidence.get("pr_url")]
    pr = evidence.get("pr")
    if isinstance(pr, dict):
        candidates.extend((pr.get("url"), pr.get("pr_url")))
    if any(isinstance(value, str) and value.rstrip("/") == pr_url.rstrip("/") for value in candidates):
        return True
    approvals = evidence.get("approvals")
    if isinstance(approvals, list):
        return any(
            isinstance(item, dict)
            and item.get("status") == "passed"
            and (
                (isinstance(item.get("pr_url"), str) and item["pr_url"].rstrip("/") == pr_url.rstrip("/"))
                or _approval_result_matches_pr(item, pr_url)
            )
            for item in approvals
        )
    return False


def _ship_pipeline_v1(
    *,
    spec: dict[str, Any],
    ctx: ExecutorContext,
    client: AgentWorkflowClient,
    agent_id: str,
    workspace_root: Path,
    env: Mapping[str, str],
    log_fn,
) -> MainExecutorResult:
    contract_version = spec.get("executor_contract_version")
    legacy_v1 = contract_version is None or contract_version == 1
    ticket_id = str(ctx.ticket.get("id") or "")
    if not ticket_id:
        return MainExecutorResult(
            status="failed",
            failure_reason="ship_pipeline_v1 requires ctx.ticket.id",
        )

    # Defense in depth: an authored hook may mark pr.approved optional, but a
    # negative approval observation must never be ignored by the mutating merge
    # executor. Routing normally prevents entry into MERGE after a negative
    # decision; this guard protects against a malformed or legacy contract.
    approval_results = [
        result for result in ctx.prior_step_results
        if isinstance(result, dict) and result.get("executor_id") == "pr.approved"
    ]
    pr_links = [str(value) for value in (ctx.ticket.get("pr_links") or []) if isinstance(value, str) and value.strip()]
    approvals_pass = bool(approval_results) and approval_results[-1].get("status") == "passed"
    if pr_links:
        approvals_pass = all(
            any(result.get("status") == "passed" and _approval_result_matches_pr(result, pr_url) for result in approval_results)
            for pr_url in pr_links
        )
    if not approvals_pass:
        if legacy_v1:
            return MainExecutorResult(
                status="failed",
                evidence={"merge_outcome": "block_merge", "merge_blocker": "approval_required"},
                failure_reason="ship_pipeline_v1 refused to merge without a passing pr.approved result",
            )
        return MainExecutorResult(
            status="completed",
            evidence={
                "evidence": {"merge": {
                    "outcome": "block_merge",
                    "final_stage": "APPROVAL",
                    "governance_exceptions": [],
                }},
            },
            failure_reason="ship_pipeline_v1 refused to merge without a passing pr.approved result",
        )

    repo = str(ctx.ticket.get("repo") or "").strip()
    ship_env = dict(env)
    ship_env.update(ctx.env)
    if repo:
        try:
            credential_environment = resolve_github_credential_environment(
                client=client,
                lease_id=ctx.lease_id,
                agent_id=agent_id,
                expected_repo=repo,
                access="write",
                base_env=ship_env,
            )
        except GitHubCredentialEnvironmentError as err:
            return MainExecutorResult(
                status="failed",
                evidence={"credential_provider": "github_app", "node_outcome": "failure"},
                failure_reason=f"ship_pipeline_v1 GitHub credential unavailable: {err}",
            )
        ship_env = credential_environment.env

    report = ship(
        ticket_id,
        client=client,
        agent_id=agent_id,
        workspace_root=workspace_root,
        env=ship_env,
        log_fn=log_fn,
        ticket_snapshot=dict(ctx.ticket),
        persist_backend=False,
    )
    ship_report = dict(report.evidence)
    merge_outcome = _merge_outcome_for_report(report.status_after, ship_report)
    evidence: dict[str, Any] = {
        "evidence": {"merge": {
        "outcome": merge_outcome,
        "final_stage": report.final_stage,
        **({"merge_commit": report.merge_commit} if report.merge_commit is not None else {}),
        "governance_exceptions": _approval_governance_exceptions(approval_results),
        }},
        "ship_report": ship_report,
    }
    if report.failure_reason is not None:
        evidence["ship_report"]["failure_reason"] = report.failure_reason
    if legacy_v1:
        legacy = dict(ship_report)
        legacy["merge_outcome"] = merge_outcome
        if report.merge_commit is not None:
            legacy["merge_commit"] = report.merge_commit
        if report.failure_reason is not None:
            legacy["failure_reason"] = report.failure_reason
        legacy["final_stage"] = report.final_stage
        legacy["node_outcome"] = "success" if merge_outcome == "completed" else "failure"
        return MainExecutorResult(status="completed", evidence=legacy)
    return MainExecutorResult(status="completed", evidence=evidence)


def _approval_governance_exceptions(results: list[dict[str, Any]]) -> list[str]:
    exceptions: list[str] = []
    for result in results:
        evidence = result.get("evidence")
        if not isinstance(evidence, dict):
            continue
        raw = evidence.get("governance_exceptions")
        if isinstance(raw, list):
            exceptions.extend(str(item) for item in raw if isinstance(item, str) and item)
    return list(dict.fromkeys(exceptions))


def _skip_validation_v1(
    *,
    spec: dict[str, Any],
    ctx: ExecutorContext,
) -> MainExecutorResult:
    del ctx
    params = spec.get("params")
    reason = "chore_skips_validation"
    if isinstance(params, dict) and isinstance(params.get("reason"), str) and params["reason"]:
        reason = params["reason"]
    return MainExecutorResult(
        status="completed",
        evidence={
            "skip_validation_outcome": "completed",
            "skip_validation_reason": reason,
            "node_outcome": "success",
        },
    )


def _merge_outcome_for_report(status_after: str | None, evidence: dict[str, object]) -> str:
    if status_after == "DEPLOY_FAILED" and _is_infra_deploy_failure(evidence):
        return "deploy_infra_failed"
    return _merge_outcome_for_status(status_after)


def _merge_outcome_for_status(status_after: str | None) -> str:
    match status_after:
        case "COMPLETED":
            return "completed"
        case "BLOCK_MERGE":
            return "block_merge"
        case "DEPLOY_FAILED":
            return "deploy_failed"
        case "PDC_FAILED":
            return "pdc_failed"
        case _:
            return "unknown"


def _is_infra_deploy_failure(evidence: dict[str, object]) -> bool:
    deploy_failure = evidence.get("deploy_failure")
    if not isinstance(deploy_failure, dict):
        return False
    deploy = deploy_failure.get("deploy")
    reason = ""
    if isinstance(deploy, dict):
        reason = str(deploy.get("reason") or deploy.get("stderr_tail") or "")
    if not reason:
        reason = str(deploy_failure.get("reason") or "")
    lowered = reason.lower()
    if re.search(r"(?<!\d)403(?!\d)", lowered):
        return True
    return any(
        token in lowered
        for token in (
            "forbidden",
            "unauthorized",
            "deploy_signal_missing",
            "missing deploy signal",
            "missing cloudflare",
            "required secrets",
        )
    )
