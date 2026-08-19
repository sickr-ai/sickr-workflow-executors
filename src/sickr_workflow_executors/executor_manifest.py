"""Canonical metadata and version checks for built-in workflow executors.

The Python descriptors are executable registrations, not a vendored copy of
the npm fixture. Release CI fetches the exact pinned contract package and
compares its fixture with this registry before building the zipapp.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping


EXECUTOR_MANIFEST_SCHEMA_VERSION = "sickr.executor_manifest.v1"
BUILTIN_EXECUTOR_FIXTURE_SCHEMA_VERSION = "sickr.builtin_executor_manifest.v1"
EXPECTED_BUILTIN_EXECUTOR_FIXTURE_SHA256 = (
    "25e6d9d8f04137a0582f0b534f39dc870e8e82096fbbec649f533f61104ffbb3"
)

ExecutorPhase = Literal["preflight", "main", "postflight"]
# Mirrors `requires` in @sickr/workflow-contracts. `partial` marks a tool list
# that is a floor rather than a total.
ExecutorRequires = Mapping[str, Any]
ExecutorKind = Literal["check", "action", "evidence", "notification", "transform"]
RiskLevel = Literal["low", "medium", "high"]

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
_PHASE_ORDER = {"preflight": 0, "main": 1, "postflight": 2}
_KINDS = {"check", "action", "evidence", "notification", "transform"}
_RISK_LEVELS = {"low", "medium", "high"}
_CUSTOM_EXECUTOR_SOURCES = {"org_repo", "sickr_default"}


class ExecutorManifestRegistrationError(ValueError):
    """Raised when registrations or the distributed fixture are inconsistent."""


@dataclass(frozen=True)
class ExecutorManifestDescriptor:
    id: str
    executor_contract_version: int
    domain_id: str
    category: str
    kind: ExecutorKind
    supported_phases: tuple[ExecutorPhase, ...]
    parameter_schema: dict[str, Any]
    output_schema: dict[str, Any]
    default_timeout_seconds: int
    mutating: bool
    risk_level: RiskLevel
    requires_workspace: bool
    source_path: str
    source_version: str | None = None
    # What the runtime must supply. None means undeclared, which is a different
    # claim from a declared-empty requirement — see the contracts package.
    # Observed by tests/_executor_probe.py, never inferred from the source.
    requires: "ExecutorRequires | None" = None
    # Older immutable workflow publications may continue to reference these
    # versions. They share the registered operation but retain version-specific
    # evidence shaping in the handler.
    compatible_contract_versions: tuple[int, ...] = ()

    def as_manifest(self) -> dict[str, Any]:
        source: dict[str, Any] = {
            "source": "sickr_default",
            "path": self.source_path,
        }
        if self.source_version is not None:
            source["version"] = self.source_version
        return {
            "schema_version": EXECUTOR_MANIFEST_SCHEMA_VERSION,
            "id": self.id,
            "executor_contract_version": self.executor_contract_version,
            "domain_id": self.domain_id,
            "category": self.category,
            "kind": self.kind,
            "supported_phases": list(self.supported_phases),
            "parameter_schema": self.parameter_schema,
            "output_schema": self.output_schema,
            "default_timeout_seconds": self.default_timeout_seconds,
            "risk": {
                "mutating": self.mutating,
                "level": self.risk_level,
                "requires_workspace": self.requires_workspace,
            },
            "source": source,
            **({"requires": dict(self.requires)} if self.requires is not None else {}),
        }


@dataclass(frozen=True)
class ExecutorReferenceError:
    path: str
    executor_id: str
    reason: str
    expected_version: int | None = None
    actual_version: Any = None

    def as_evidence(self, workflow_contract_version: Any) -> dict[str, Any]:
        evidence = {
            "path": self.path,
            "executor_id": self.executor_id,
            "reason": self.reason,
            "workflow_contract_version": workflow_contract_version,
        }
        if self.expected_version is not None:
            evidence["expected_executor_contract_version"] = self.expected_version
        if self.actual_version is not None:
            evidence["actual_executor_contract_version"] = self.actual_version
        return evidence


def build_executor_manifest_registry(
    descriptors: Iterable[ExecutorManifestDescriptor],
) -> dict[str, ExecutorManifestDescriptor]:
    registry: dict[str, ExecutorManifestDescriptor] = {}
    for descriptor in descriptors:
        _validate_descriptor(descriptor)
        if descriptor.id in registry:
            raise ExecutorManifestRegistrationError(
                f"duplicate executor manifest id: {descriptor.id}"
            )
        registry[descriptor.id] = descriptor
    if not registry:
        raise ExecutorManifestRegistrationError(
            "executor manifest registry must not be empty"
        )
    return dict(sorted(registry.items()))


def assert_registration_conformance(
    *,
    descriptors: Mapping[str, ExecutorManifestDescriptor],
    registrations: Mapping[str, Any],
    registration_name: str,
) -> None:
    descriptor_ids = set(descriptors)
    registration_ids = set(registrations)
    missing = sorted(descriptor_ids - registration_ids)
    extra = sorted(registration_ids - descriptor_ids)
    details: list[str] = []
    if missing:
        details.append(f"missing handlers: {missing}")
    if extra:
        details.append(f"unmanifested handlers: {extra}")
    if details:
        raise ExecutorManifestRegistrationError(
            f"{registration_name} does not match executor manifests; "
            + "; ".join(details)
        )


def builtin_executor_fixture() -> dict[str, Any]:
    return {
        "schema_version": BUILTIN_EXECUTOR_FIXTURE_SCHEMA_VERSION,
        "executors": [
            descriptor.as_manifest()
            for descriptor in BUILTIN_EXECUTOR_MANIFESTS.values()
        ],
    }


def stable_serialize_executor_fixture(fixture: Any) -> str:
    return json.dumps(
        fixture,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def stable_executor_fixture_sha256(fixture: Any) -> str:
    return hashlib.sha256(
        stable_serialize_executor_fixture(fixture).encode("utf-8")
    ).hexdigest()


def verify_executor_fixture(path: Path) -> str:
    try:
        fixture = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as err:
        raise ExecutorManifestRegistrationError(
            f"cannot read executor fixture {path}: {err}"
        ) from err

    checksum = stable_executor_fixture_sha256(fixture)
    if checksum != EXPECTED_BUILTIN_EXECUTOR_FIXTURE_SHA256:
        raise ExecutorManifestRegistrationError(
            "executor fixture checksum does not match the pinned canonical "
            f"checksum: expected {EXPECTED_BUILTIN_EXECUTOR_FIXTURE_SHA256}, "
            f"got {checksum}"
        )
    expected = builtin_executor_fixture()
    if fixture != expected:
        raise ExecutorManifestRegistrationError(
            _fixture_mismatch_message(expected=expected, actual=fixture)
        )
    return checksum


def validate_contract_executor_references(
    contract: Mapping[str, Any],
) -> list[ExecutorReferenceError]:
    """Validate built-in ids and explicit per-executor versions before work.

    Missing versions remain accepted during the additive compatibility window.
    Once a version is present it must exactly match; there is no latest-version
    fallback. Org-repo/script flight executors are outside the built-in registry.
    """

    errors: list[ExecutorReferenceError] = []
    main = contract.get("main")
    if isinstance(main, dict) and main.get("executor") == "deterministic":
        operation_id = str(main.get("operation_id") or "").strip()
        errors.extend(
            _validate_reference(
                path="contract.main",
                executor_id=operation_id,
                version=main.get("executor_contract_version"),
                source=None,
                allow_custom=False,
            )
        )
        if operation_id == "executor_steps_v1":
            params = main.get("params")
            config = params.get("config_json") if isinstance(params, dict) else None
            errors.extend(
                _validate_step_config(
                    config,
                    path="contract.main.params.config_json",
                )
            )

    state_execution = contract.get("state_execution")
    hooks = (
        state_execution.get("hooks")
        if isinstance(state_execution, dict)
        else None
    )
    if isinstance(hooks, dict):
        for phase in ("preflight", "postflight"):
            hook = hooks.get(phase)
            if not isinstance(hook, dict) or hook.get("enabled") is not True:
                continue
            errors.extend(
                _validate_step_config(
                    hook.get("config_json"),
                    path=f"contract.state_execution.hooks.{phase}.config_json",
                )
            )
    return errors


def validate_step_executor_reference(
    *,
    executor_id: str,
    executor_contract_version: Any,
    source: Mapping[str, Any] | None,
) -> ExecutorReferenceError | None:
    errors = _validate_reference(
        path="executor_step",
        executor_id=executor_id,
        version=executor_contract_version,
        source=source,
        allow_custom=True,
    )
    return errors[0] if errors else None


def _validate_step_config(config: Any, *, path: str) -> list[ExecutorReferenceError]:
    if not isinstance(config, dict):
        return []
    steps = config.get("steps")
    if not isinstance(steps, list):
        return []
    errors: list[ExecutorReferenceError] = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        if step.get("enabled") is False:
            continue
        executor_id = str(step.get("executor_id") or "").strip()
        errors.extend(
            _validate_reference(
                path=f"{path}.steps[{index}]",
                executor_id=executor_id,
                version=step.get("executor_contract_version"),
                source=step.get("source") if isinstance(step.get("source"), dict) else None,
                allow_custom=True,
            )
        )
    return errors


def _validate_reference(
    *,
    path: str,
    executor_id: str,
    version: Any,
    source: Mapping[str, Any] | None,
    allow_custom: bool,
) -> list[ExecutorReferenceError]:
    descriptor = BUILTIN_EXECUTOR_MANIFESTS.get(executor_id)
    if descriptor is None:
        source_kind = source.get("source") if source is not None else None
        if allow_custom and source_kind in _CUSTOM_EXECUTOR_SOURCES:
            return []
        # Additive compatibility window: legacy contracts did not carry a
        # per-executor version and tests/plugins may inject an unversioned
        # local handler. Once a version is present, the id must be in either
        # the built-in registry or an explicitly sourced custom registry.
        if version is None:
            return []
        return [
            ExecutorReferenceError(
                path=path,
                executor_id=executor_id,
                reason="unknown built-in executor id",
                actual_version=version,
            )
        ]
    if version is None:
        return []
    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or version not in (descriptor.executor_contract_version, *descriptor.compatible_contract_versions)
    ):
        return [
            ExecutorReferenceError(
                path=path,
                executor_id=executor_id,
                reason="executor_contract_version mismatch",
                expected_version=descriptor.executor_contract_version,
                actual_version=version,
            )
        ]
    return []


def _validate_descriptor(descriptor: ExecutorManifestDescriptor) -> None:
    if not _ID_RE.fullmatch(descriptor.id):
        raise ExecutorManifestRegistrationError(
            f"invalid executor manifest id: {descriptor.id!r}"
        )
    version = descriptor.executor_contract_version
    if isinstance(version, bool) or not isinstance(version, int) or version <= 0:
        raise ExecutorManifestRegistrationError(
            f"{descriptor.id}: executor_contract_version must be a positive integer"
        )
    if not descriptor.domain_id.strip() or not descriptor.category.strip():
        raise ExecutorManifestRegistrationError(
            f"{descriptor.id}: domain_id and category must be non-empty"
        )
    if descriptor.kind not in _KINDS:
        raise ExecutorManifestRegistrationError(
            f"{descriptor.id}: invalid executor kind {descriptor.kind!r}"
        )
    if not descriptor.supported_phases:
        raise ExecutorManifestRegistrationError(
            f"{descriptor.id}: supported_phases must not be empty"
        )
    phase_ranks = [_PHASE_ORDER.get(phase) for phase in descriptor.supported_phases]
    if (
        any(rank is None for rank in phase_ranks)
        or len(set(descriptor.supported_phases)) != len(descriptor.supported_phases)
        or phase_ranks != sorted(phase_ranks)  # type: ignore[type-var]
    ):
        raise ExecutorManifestRegistrationError(
            f"{descriptor.id}: supported_phases must be unique and canonical"
        )
    for name, schema in (
        ("parameter_schema", descriptor.parameter_schema),
        ("output_schema", descriptor.output_schema),
    ):
        if not isinstance(schema, dict) or schema.get("type") != "object":
            raise ExecutorManifestRegistrationError(
                f"{descriptor.id}: {name} must be an object JSON schema"
            )
    timeout = descriptor.default_timeout_seconds
    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0:
        raise ExecutorManifestRegistrationError(
            f"{descriptor.id}: default_timeout_seconds must be positive"
        )
    if descriptor.risk_level not in _RISK_LEVELS:
        raise ExecutorManifestRegistrationError(
            f"{descriptor.id}: invalid risk level {descriptor.risk_level!r}"
        )
    if not descriptor.source_path.startswith("builtin://"):
        raise ExecutorManifestRegistrationError(
            f"{descriptor.id}: built-in source path must use builtin://"
        )


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ExecutorManifestRegistrationError(
                f"executor fixture contains duplicate JSON key {key!r}"
            )
        value[key] = item
    return value


def _fixture_mismatch_message(*, expected: Any, actual: Any) -> str:
    expected_entries = (
        expected.get("executors") if isinstance(expected, dict) else None
    )
    actual_entries = actual.get("executors") if isinstance(actual, dict) else None
    expected_versions = {
        entry.get("id"): entry.get("executor_contract_version")
        for entry in expected_entries or []
        if isinstance(entry, dict)
    }
    actual_versions = {
        entry.get("id"): entry.get("executor_contract_version")
        for entry in actual_entries or []
        if isinstance(entry, dict)
    }
    return (
        "executor fixture does not match the Python registry; "
        f"expected ids/versions={expected_versions}, "
        f"actual ids/versions={actual_versions}"
    )


BUILTIN_EXECUTOR_MANIFESTS = build_executor_manifest_registry(
    (
        ExecutorManifestDescriptor(
            id="actor.completed",
            executor_contract_version=1,
            domain_id="core",
            category="Evidence",
            kind="evidence",
            supported_phases=("postflight",),
            parameter_schema={"type": "object", "properties": {}},
            output_schema={
                "type": "object",
                "properties": {"status": {"type": "string"}},
            },
            default_timeout_seconds=120,
            mutating=False,
            risk_level="low",
            requires_workspace=False,
            source_path="builtin://executors/actor.completed",
            requires={"connections": [], "tools": []},
        ),
        ExecutorManifestDescriptor(
            id="ci.run_build",
            executor_contract_version=1,
            domain_id="software-engineering",
            category="Validation",
            kind="check",
            supported_phases=("preflight", "postflight"),
            parameter_schema={
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "commands": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "working_directory": {"type": "string"},
                },
            },
            output_schema={
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "command_kind": {"type": "string"},
                    "working_directory": {"type": "string"},
                    "exit_code": {"type": "integer"},
                    "stdout": {"type": "string"},
                    "stderr": {"type": "string"},
                    "commands": {"type": "array"},
                },
            },
            default_timeout_seconds=1200,
            mutating=True,
            risk_level="high",
            requires_workspace=True,
            source_path="builtin://executors/ci.run_build",
            requires={"connections": [], "tools": []},
        ),
        ExecutorManifestDescriptor(
            id="ci.run_command",
            executor_contract_version=1,
            domain_id="software-engineering",
            category="Validation",
            kind="check",
            supported_phases=("preflight", "postflight"),
            parameter_schema={
                "type": "object",
                "properties": {
                    "step_name": {"type": "string"},
                    "command": {"type": "string"},
                    "commands": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "working_directory": {"type": "string"},
                },
                "required": ["step_name"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "step_name": {"type": "string"},
                    "command": {"type": "string"},
                    "command_kind": {"type": "string"},
                    "working_directory": {"type": "string"},
                    "exit_code": {"type": "integer"},
                    "stdout": {"type": "string"},
                    "stderr": {"type": "string"},
                    "commands": {"type": "array"},
                },
            },
            default_timeout_seconds=900,
            mutating=True,
            risk_level="high",
            requires_workspace=True,
            source_path="builtin://executors/ci.run_command",
            requires={"connections": [], "tools": []},
        ),
        ExecutorManifestDescriptor(
            id="ci.run_tests",
            executor_contract_version=1,
            domain_id="software-engineering",
            category="Validation",
            kind="check",
            supported_phases=("preflight", "postflight"),
            parameter_schema={
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "commands": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "working_directory": {"type": "string"},
                },
            },
            output_schema={
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "command_kind": {"type": "string"},
                    "working_directory": {"type": "string"},
                    "exit_code": {"type": "integer"},
                    "stdout": {"type": "string"},
                    "stderr": {"type": "string"},
                    "commands": {"type": "array"},
                },
            },
            default_timeout_seconds=900,
            mutating=True,
            risk_level="high",
            requires_workspace=True,
            source_path="builtin://executors/ci.run_tests",
            requires={"connections": [], "tools": []},
        ),
        ExecutorManifestDescriptor(
            id="ci.run_typecheck",
            executor_contract_version=1,
            domain_id="software-engineering",
            category="Validation",
            kind="check",
            supported_phases=("preflight", "postflight"),
            parameter_schema={
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "commands": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "working_directory": {"type": "string"},
                },
            },
            output_schema={
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "command_kind": {"type": "string"},
                    "working_directory": {"type": "string"},
                    "exit_code": {"type": "integer"},
                    "stdout": {"type": "string"},
                    "stderr": {"type": "string"},
                    "commands": {"type": "array"},
                },
            },
            default_timeout_seconds=900,
            mutating=True,
            risk_level="high",
            requires_workspace=True,
            source_path="builtin://executors/ci.run_typecheck",
            requires={"connections": [], "tools": []},
        ),
        ExecutorManifestDescriptor(
            id="create_tickets_v1",
            executor_contract_version=1,
            domain_id="core",
            category="Ticketing",
            kind="action",
            supported_phases=("main",),
            parameter_schema={
                "type": "object",
                "properties": {"plan": {"type": "object"}},
            },
            output_schema={
                "type": "object",
                "properties": {
                    "created_tickets": {"type": "array"},
                    "created_ticket_ids": {"type": "array"},
                    "node_outcome": {"type": "string"},
                },
            },
            default_timeout_seconds=600,
            mutating=True,
            risk_level="high",
            requires_workspace=False,
            source_path="builtin://operations/create_tickets_v1",
        ),
        ExecutorManifestDescriptor(
            id="channel.require",
            executor_contract_version=1,
            domain_id="core",
            category="Channels",
            kind="evidence",
            supported_phases=("preflight", "postflight"),
            parameter_schema={
                "type": "object",
                "properties": {"channel": {"type": "string"}},
                "required": ["channel"],
                "additionalProperties": False,
            },
            output_schema={
                "type": "object",
                "properties": {"channel": {"type": "string"}, "root": {"type": "string"}},
            },
            default_timeout_seconds=120,
            mutating=False,
            risk_level="low",
            requires_workspace=False,
            source_path="builtin://executors/channel.require",
            requires={"connections": [], "tools": []},
        ),
        ExecutorManifestDescriptor(
            id="evidence.has_any",
            executor_contract_version=1,
            domain_id="core",
            category="Evidence",
            kind="evidence",
            supported_phases=("preflight", "postflight"),
            parameter_schema={
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "paths": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["paths"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "matched_path": {"type": "string"},
                },
            },
            default_timeout_seconds=120,
            mutating=False,
            risk_level="low",
            requires_workspace=False,
            source_path="builtin://executors/evidence.has_any",
            requires={"connections": [], "tools": []},
        ),
        ExecutorManifestDescriptor(
            id="executor_steps_v1",
            executor_contract_version=1,
            domain_id="core",
            category="Composition",
            kind="action",
            supported_phases=("main",),
            parameter_schema={
                "type": "object",
                "properties": {"config_json": {"type": "object"}},
            },
            output_schema={
                "type": "object",
                "properties": {
                    "steps": {"type": "array"},
                    "node_outcome": {"type": "string"},
                },
            },
            default_timeout_seconds=600,
            mutating=True,
            risk_level="high",
            requires_workspace=False,
            source_path="builtin://operations/executor_steps_v1",
        ),
        ExecutorManifestDescriptor(
            id="runtime.noop",
            executor_contract_version=1,
            domain_id="core",
            category="Runtime",
            kind="action",
            supported_phases=("main",),
            parameter_schema={"type": "object", "additionalProperties": False},
            output_schema={"type": "object", "properties": {}},
            default_timeout_seconds=30,
            mutating=False,
            risk_level="low",
            requires_workspace=False,
            source_path="builtin://operations/runtime.noop",
        ),
        ExecutorManifestDescriptor(
            id="git.commit_state_changes",
            executor_contract_version=1,
            domain_id="software-engineering",
            category="Repository",
            kind="action",
            supported_phases=("postflight",),
            parameter_schema={
                "type": "object",
                "properties": {
                    "working_directory": {"type": "string"},
                    "message": {"type": "string"},
                },
            },
            output_schema={
                "type": "object",
                "properties": {
                    "committed": {"type": "boolean"},
                    "head_sha": {"type": "string"},
                    "changed_status": {"type": "string"},
                },
            },
            default_timeout_seconds=120,
            mutating=True,
            risk_level="high",
            requires_workspace=True,
            source_path="builtin://executors/git.commit_state_changes",
            requires={"connections": [], "tools": ["git"]},
        ),
        ExecutorManifestDescriptor(
            id="package.install_dependencies",
            executor_contract_version=1,
            domain_id="software-engineering",
            category="Setup",
            kind="action",
            supported_phases=("preflight",),
            parameter_schema={"type": "object", "properties": {"command": {"type": "string"}, "commands": {"type": "array", "items": {"type": "string"}}, "working_directory": {"type": "string"}}},
            output_schema={"type": "object", "properties": {"command": {"type": "string"}, "exit_code": {"type": "integer"}, "commands": {"type": "array"}}},
            default_timeout_seconds=1200,
            mutating=True,
            risk_level="medium",
            requires_workspace=True,
            source_path="builtin://executors/package.install_dependencies",
            requires={"connections": [], "tools": []},
        ),
        ExecutorManifestDescriptor(
            id="workspace.clone_repositories",
            executor_contract_version=1,
            domain_id="software-engineering",
            category="Setup",
            kind="action",
            supported_phases=("preflight",),
            parameter_schema={"type": "object", "properties": {}},
            output_schema={"type": "object", "properties": {"workspace_path": {"type": "string"}, "remote_url": {"type": "string"}, "cloned": {"type": "boolean"}, "repositories": {"type": "array"}}},
            default_timeout_seconds=300,
            mutating=True,
            risk_level="medium",
            requires_workspace=False,
            source_path="builtin://executors/workspace.clone_repositories",
            requires={"connections": ["github"], "tools": ["git"]},
        ),
        ExecutorManifestDescriptor(
            id="validation.run_baseline",
            executor_contract_version=1,
            domain_id="software-engineering",
            category="Setup",
            kind="check",
            supported_phases=("preflight",),
            parameter_schema={"type": "object", "properties": {"command": {"type": "string"}, "commands": {"type": "array", "items": {"type": "string"}}, "working_directory": {"type": "string"}}},
            output_schema={"type": "object", "properties": {"command": {"type": "string"}, "exit_code": {"type": "integer"}, "commands": {"type": "array"}}},
            default_timeout_seconds=1200,
            mutating=True,
            risk_level="medium",
            requires_workspace=True,
            source_path="builtin://executors/validation.run_baseline",
            requires={"connections": [], "tools": []},
        ),
        ExecutorManifestDescriptor(
            id="workspace.capture_generated_ignores",
            executor_contract_version=1,
            domain_id="software-engineering",
            category="Setup",
            kind="action",
            supported_phases=("preflight",),
            parameter_schema={"type": "object", "properties": {"working_directory": {"type": "string"}}},
            output_schema={"type": "object", "properties": {"excluded_paths": {"type": "array"}, "exclude_file": {"type": "string"}}},
            default_timeout_seconds=120,
            mutating=True,
            risk_level="low",
            requires_workspace=True,
            source_path="builtin://executors/workspace.capture_generated_ignores",
            requires={"connections": [], "tools": ["git"]},
        ),
        ExecutorManifestDescriptor(
            id="git.install_push_guard",
            executor_contract_version=1,
            domain_id="software-engineering",
            category="Setup",
            kind="action",
            supported_phases=("preflight",),
            parameter_schema={
                "type": "object",
                "required": ["protected_branches"],
                "properties": {
                    "protected_branches": {
                        "type": "array",
                        "minItems": 1,
                        "uniqueItems": True,
                        "items": {"type": "string", "minLength": 1},
                    },
                    "working_directory": {"type": "string"},
                },
            },
            output_schema={
                "type": "object",
                "properties": {
                    "protected_branches": {"type": "array", "items": {"type": "string"}},
                    "hook_path": {"type": "string"},
                },
            },
            default_timeout_seconds=30,
            mutating=True,
            risk_level="low",
            requires_workspace=True,
            source_path="builtin://executors/git.install_push_guard",
            requires={"connections": [], "tools": ["git"]},
        ),
        ExecutorManifestDescriptor(
            id="git.ensure_published",
            executor_contract_version=1,
            domain_id="software-engineering",
            category="Repository",
            kind="action",
            supported_phases=("postflight",),
            parameter_schema={
                "type": "object",
                "properties": {
                    "base_branch": {"type": "string"},
                    "target_branch": {"type": "string"},
                    "working_directory": {"type": "string"},
                },
            },
            output_schema={
                "type": "object",
                "properties": {
                    "base_branch": {"type": "string"},
                    "target_branch": {"type": "string"},
                    "current_branch": {"type": "string"},
                    "head_sha": {"type": "string"},
                    "remote_head_sha": {"type": "string"},
                    "pushed": {"type": "boolean"},
                },
            },
            default_timeout_seconds=300,
            mutating=True,
            risk_level="high",
            requires_workspace=True,
            source_path="builtin://executors/git.ensure_published",
            requires={"connections": ["github"], "tools": ["git"]},
        ),
        ExecutorManifestDescriptor(
            id="git.ensure_ticket_branch",
            executor_contract_version=1,
            domain_id="software-engineering",
            category="Repository",
            kind="action",
            supported_phases=("preflight",),
            parameter_schema={
                "type": "object",
                "properties": {
                    "base_branch": {"type": "string"},
                    "target_branch": {"type": "string"},
                    "working_directory": {"type": "string"},
                },
            },
            output_schema={
                "type": "object",
                "properties": {
                    "base_branch": {"type": "string"},
                    "target_branch": {"type": "string"},
                    "branch": {"type": "string"},
                    "before_head_sha": {"type": "string"},
                    "after_head_sha": {"type": "string"},
                    "created": {"type": "boolean"},
                    "fast_forwarded": {"type": "boolean"},
                },
            },
            default_timeout_seconds=300,
            mutating=True,
            risk_level="medium",
            requires_workspace=True,
            source_path="builtin://executors/git.ensure_ticket_branch",
            requires={"connections": ["github"], "tools": ["git"]},
        ),
        ExecutorManifestDescriptor(
            id="git.sync_with_base",
            executor_contract_version=1,
            domain_id="software-engineering",
            category="Repository",
            kind="action",
            supported_phases=("preflight",),
            parameter_schema={
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["merge", "rebase"],
                    },
                    "base_branch": {"type": "string"},
                    "target_branch": {"type": "string"},
                    "working_directory": {"type": "string"},
                },
            },
            output_schema={
                "type": "object",
                "properties": {
                    "mode": {"type": "string"},
                    "base_branch": {"type": "string"},
                    "target_branch": {"type": "string"},
                    "before_head_sha": {"type": "string"},
                    "base_sha": {"type": "string"},
                    "after_head_sha": {"type": "string"},
                    "updated": {"type": "boolean"},
                    "failure_stage": {"type": "string"},
                },
            },
            default_timeout_seconds=300,
            mutating=True,
            risk_level="high",
            requires_workspace=True,
            source_path="builtin://executors/git.sync_with_base",
            requires={"connections": ["github"], "tools": ["git"]},
        ),
        ExecutorManifestDescriptor(
            id="github.ensure_deployed",
            executor_contract_version=1,
            domain_id="software-engineering",
            category="Deployment",
            kind="check",
            supported_phases=("postflight",),
            parameter_schema={
                "type": "object",
                "properties": {
                    "health_url": {"type": "string"},
                    "commit_field": {"type": "string"},
                    "branch": {"type": "string"},
                    "poll_seconds": {"type": "integer"},
                    "poll_interval_seconds": {"type": "integer"},
                    "retrigger": {"type": "boolean"},
                    "probe_timeout_seconds": {"type": "integer"},
                    "working_directory": {"type": "string"},
                },
                "required": ["health_url"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "expected_sha": {"type": "string"},
                    "deployed_sha": {"type": "string"},
                    "retriggered": {"type": "boolean"},
                    "poll_attempts": {"type": "integer"},
                },
            },
            default_timeout_seconds=1800,
            mutating=True,
            risk_level="high",
            requires_workspace=True,
            source_path="builtin://executors/github.ensure_deployed",
            requires={"connections": ["github"], "tools": ["git"], "partial": True},
        ),
        ExecutorManifestDescriptor(
            id="github.prepare_pr_workspace",
            executor_contract_version=1,
            domain_id="software-engineering",
            category="Pull Request",
            kind="action",
            supported_phases=("preflight",),
            parameter_schema={
                "type": "object",
                "properties": {"working_directory": {"type": "string"}},
            },
            output_schema={
                "type": "object",
                "properties": {
                    "repo": {"type": "string"},
                    "pr_url": {"type": "string"},
                    "pr_number": {"type": "integer"},
                    "base_branch": {"type": "string"},
                    "base_sha": {"type": "string"},
                    "head_branch": {"type": "string"},
                    "head_sha": {"type": "string"},
                    "merge_base_sha": {"type": "string"},
                    "diff_range": {"type": "string"},
                    "changed_files": {"type": "array"},
                },
            },
            default_timeout_seconds=300,
            mutating=True,
            risk_level="medium",
            requires_workspace=True,
            source_path="builtin://executors/github.prepare_pr_workspace",
            requires={"connections": ["github"], "tools": ["gh", "git"], "partial": True},
        ),
        ExecutorManifestDescriptor(
            id="github.publish_and_ensure_pr",
            executor_contract_version=1,
            domain_id="software-engineering",
            category="Pull Request",
            kind="action",
            supported_phases=("postflight",),
            parameter_schema={
                "type": "object",
                "properties": {
                    "working_directory": {"type": "string"},
                    "title": {"type": "string"},
                    "body": {"type": "string"},
                },
            },
            output_schema={
                "type": "object",
                "properties": {
                    "repo": {"type": "string"},
                    "base_branch": {"type": "string"},
                    "target_branch": {"type": "string"},
                    "published": {"type": "boolean"},
                    "pr_url": {"type": "string"},
                    "pr_number": {"type": "integer"},
                    "created": {"type": "boolean"},
                    "state": {"type": "string"},
                },
            },
            default_timeout_seconds=300,
            mutating=True,
            risk_level="high",
            requires_workspace=True,
            source_path="builtin://executors/github.publish_and_ensure_pr",
            requires={"connections": ["github"], "tools": ["git"], "partial": True},
        ),
        ExecutorManifestDescriptor(
            id="github.sync_review_decision",
            executor_contract_version=1,
            domain_id="software-engineering",
            category="Pull Request",
            kind="action",
            supported_phases=("postflight",),
            parameter_schema={
                "type": "object",
                "properties": {
                    "channel": {"type": "string"},
                    "decision": {"type": "string"},
                    "body": {"type": "string"},
                },
            },
            output_schema={
                "type": "object",
                "properties": {
                    "repo": {"type": "string"},
                    "pr_number": {"type": "integer"},
                    "pr_url": {"type": "string"},
                    "sickr_review_decision": {"type": "string"},
                    "requested_github_review_event": {"type": "string"},
                    "github_review_event": {"type": "string"},
                    "delivery_mode": {"type": "string"},
                    "formal_review": {"type": "boolean"},
                    "review_url": {"type": "string"},
                },
            },
            default_timeout_seconds=120,
            mutating=True,
            risk_level="medium",
            requires_workspace=False,
            source_path="builtin://executors/github.sync_review_decision",
            requires={"connections": ["github"], "tools": [], "partial": True},
        ),
        ExecutorManifestDescriptor(
            id="sources.collect_v1",
            executor_contract_version=1,
            domain_id="executive-intelligence",
            category="Research",
            kind="action",
            supported_phases=("main",),
            parameter_schema={
                "type": "object",
                "properties": {
                    "provider_id": {"type": "string"},
                    "query": {"type": "object"},
                    "cursor": {"type": ["string", "null"]},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                },
                "required": ["provider_id", "query", "limit"],
            },
            output_schema={"type": "object", "properties": {"source_records": {"type": "array"}, "provider_evidence": {"type": "object"}}},
            default_timeout_seconds=120,
            mutating=False,
            risk_level="medium",
            requires_workspace=False,
            source_path="builtin://executors/sources.collect_v1",
        ),
        ExecutorManifestDescriptor(
            id="records.normalize_v1",
            executor_contract_version=1,
            domain_id="core",
            category="Records",
            kind="transform",
            supported_phases=("main",),
            parameter_schema={
                "type": "object",
                "properties": {
                    "input_path": {"type": "string"},
                    "field_mapping": {"type": "object"},
                    "required_fields": {"type": "array", "items": {"type": "string"}},
                    "dedupe_keys": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["input_path"],
            },
            output_schema={"type": "object", "properties": {"records": {"type": "array"}, "rejected_records": {"type": "array"}}},
            default_timeout_seconds=120,
            mutating=False,
            risk_level="low",
            requires_workspace=False,
            source_path="builtin://executors/records.normalize_v1",
        ),
        ExecutorManifestDescriptor(
            id="records.rank_v1",
            executor_contract_version=1,
            domain_id="core",
            category="Records",
            kind="transform",
            supported_phases=("main",),
            parameter_schema={
                "type": "object",
                "properties": {
                    "input_path": {"type": "string"},
                    "criteria": {"type": "array"},
                    "tie_breakers": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["input_path", "criteria"],
            },
            output_schema={"type": "object", "properties": {"ranked_records": {"type": "array"}}},
            default_timeout_seconds=120,
            mutating=False,
            risk_level="low",
            requires_workspace=False,
            source_path="builtin://executors/records.rank_v1",
        ),
        ExecutorManifestDescriptor(
            id="records.select_v1",
            executor_contract_version=1,
            domain_id="core",
            category="Records",
            kind="transform",
            supported_phases=("main",),
            parameter_schema={
                "type": "object",
                "properties": {
                    "input_path": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                    "minimum_score": {"type": "number"},
                },
                "required": ["input_path", "limit", "minimum_score"],
            },
            output_schema={"type": "object", "properties": {"selected_records": {"type": "array"}, "empty_result": {"type": "boolean"}}},
            default_timeout_seconds=120,
            mutating=False,
            risk_level="low",
            requires_workspace=False,
            source_path="builtin://executors/records.select_v1",
        ),
        ExecutorManifestDescriptor(
            id="signals.ingest_ops_v1",
            executor_contract_version=1,
            domain_id="executive-intelligence",
            category="Signals",
            kind="action",
            supported_phases=("main",),
            parameter_schema={
                "type": "object",
                "properties": {"input_path": {"type": "string"}},
                "required": ["input_path"],
            },
            output_schema={"type": "object", "properties": {"ops_signal_radar_ingest": {"type": "object"}}},
            default_timeout_seconds=120,
            mutating=True,
            risk_level="medium",
            requires_workspace=False,
            source_path="builtin://executors/signals.ingest_ops_v1",
        ),
        ExecutorManifestDescriptor(
            id="integration.dispatch",
            executor_contract_version=2,
            domain_id="core",
            category="Integrations",
            kind="notification",
            supported_phases=("preflight", "postflight"),
            parameter_schema={
                "type": "object",
                "properties": {
                    "provider": {
                        "type": "string",
                        "title": "Provider",
                        "x_sickr_ui": {
                            "control": "select",
                            "allow_custom": True,
                            "options": {"source": "integration.providers"},
                        },
                    },
                    "action": {
                        "type": "string",
                        "title": "Action",
                        "x_sickr_ui": {
                            "control": "select",
                            "allow_custom": True,
                            "options": {
                                "source": "integration.actions",
                                "parameters": {"provider": "$params.provider"},
                            },
                        },
                    },
                    "payload": {"type": "object"},
                },
                "required": ["provider", "action"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "provider": {"type": "string"},
                    "action": {"type": "string"},
                    "slack_channel_id": {"type": "string"},
                    "slack_message_ts": {"type": "string"},
                },
            },
            default_timeout_seconds=120,
            mutating=True,
            risk_level="medium",
            requires_workspace=False,
            source_path="builtin://executors/integration.dispatch",
            requires={"connections": ["integration"], "tools": [], "partial": True},
        ),
        ExecutorManifestDescriptor(
            id="pr.approved",
            executor_contract_version=1,
            domain_id="software-engineering",
            category="Pull Request",
            kind="check",
            supported_phases=("preflight", "postflight"),
            parameter_schema={"type": "object", "properties": {}},
            output_schema={
                "type": "object",
                "properties": {
                    "reviewDecision": {"type": "string"},
                    "reviews": {"type": "array"},
                },
            },
            default_timeout_seconds=120,
            mutating=False,
            risk_level="low",
            requires_workspace=False,
            source_path="builtin://executors/pr.approved",
            requires={"connections": ["github"], "tools": ["gh"]},
        ),
        ExecutorManifestDescriptor(
            id="pr.checks_green",
            executor_contract_version=1,
            domain_id="software-engineering",
            category="Pull Request",
            kind="check",
            supported_phases=("preflight", "postflight"),
            parameter_schema={
                "type": "object",
                "properties": {
                    "wait_timeout_seconds": {"type": "integer"},
                    "poll_interval_seconds": {"type": "integer"},
                },
            },
            output_schema={
                "type": "object",
                "properties": {
                    "statusCheckRollup": {"type": "array"},
                    "waited_for_checks": {"type": "boolean"},
                },
            },
            default_timeout_seconds=600,
            mutating=False,
            risk_level="low",
            requires_workspace=False,
            source_path="builtin://executors/pr.checks_green",
            requires={"connections": ["github"], "tools": ["gh"], "partial": True},
        ),
        ExecutorManifestDescriptor(
            id="pr.extract_context",
            executor_contract_version=1,
            domain_id="software-engineering",
            category="Pull Request",
            kind="evidence",
            supported_phases=("preflight",),
            parameter_schema={
                "type": "object",
                "properties": {"required": {"type": "boolean"}},
            },
            output_schema={
                "type": "object",
                "properties": {
                    "pr": {"type": "object"},
                    "pr_links": {"type": "array"},
                    "implementation_author": {"type": "string"},
                },
            },
            default_timeout_seconds=120,
            mutating=False,
            risk_level="low",
            requires_workspace=False,
            source_path="builtin://executors/pr.extract_context",
            requires={"connections": ["github"], "tools": ["gh"]},
        ),
        ExecutorManifestDescriptor(
            id="pr.mergeable",
            executor_contract_version=1,
            domain_id="software-engineering",
            category="Pull Request",
            kind="check",
            supported_phases=("preflight",),
            parameter_schema={"type": "object", "properties": {}},
            output_schema={
                "type": "object",
                "properties": {
                    "mergeStateStatus": {"type": "string"},
                    "isDraft": {"type": "boolean"},
                    "state": {"type": "string"},
                },
            },
            default_timeout_seconds=120,
            mutating=False,
            risk_level="low",
            requires_workspace=False,
            source_path="builtin://executors/pr.mergeable",
            requires={"connections": ["github"], "tools": ["gh"]},
        ),
        ExecutorManifestDescriptor(
            id="remediation.spawn_tickets",
            executor_contract_version=1,
            domain_id="core",
            category="Ticketing",
            kind="action",
            supported_phases=("postflight",),
            parameter_schema={
                "type": "object",
                "properties": {
                    "spawn_status": {
                        "type": "string",
                        "enum": ["DRAFT", "READY"],
                    },
                    "max_cycles": {"type": "integer"},
                },
            },
            output_schema={
                "type": "object",
                "properties": {"remediation": {"type": "object"}},
            },
            default_timeout_seconds=120,
            mutating=True,
            risk_level="high",
            requires_workspace=False,
            source_path="builtin://executors/remediation.spawn_tickets",
            requires={"connections": [], "tools": []},
        ),
        ExecutorManifestDescriptor(
            id="script_worker_v1",
            executor_contract_version=1,
            domain_id="core",
            category="Custom code",
            kind="action",
            supported_phases=("main",),
            parameter_schema={
                "type": "object",
                "properties": {
                    "configured_code_source": {"type": "object"},
                    "effective_code_source": {"type": "object"},
                    "config_json": {"type": "object"},
                },
            },
            output_schema={
                "type": "object",
                "properties": {
                    "node_outcome": {"type": "string"},
                    "source_provenance": {"type": "object"},
                },
            },
            default_timeout_seconds=600,
            mutating=True,
            risk_level="high",
            requires_workspace=True,
            source_path="builtin://operations/script_worker_v1",
        ),
        ExecutorManifestDescriptor(
            id="ship_pipeline_v1",
            executor_contract_version=2,
            domain_id="software-engineering",
            category="Delivery",
            kind="action",
            supported_phases=("main",),
            parameter_schema={"type": "object", "properties": {}},
            output_schema={
                "type": "object",
                "properties": {
                    "evidence": {
                        "type": "object",
                        "properties": {"merge": {
                            "type": "object",
                            "properties": {
                                "outcome": {"type": "string", "enum": ["completed", "block_merge", "deploy_failed", "deploy_infra_failed", "pdc_failed", "unknown"]},
                                "merge_commit": {"type": "string"},
                                "final_stage": {"type": "string"},
                                "governance_exceptions": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": ["outcome", "final_stage"],
                        }},
                        "required": ["merge"],
                    },
                    "ship_report": {"type": "object"},
                },
                "required": ["evidence"],
                "x-sickr-output-channels": [
                    {
                        "canonical_path": "evidence.merge.outcome", "label": "Merge outcome",
                        "description": "Final deterministic disposition of the shipping pipeline.",
                        "value_type": "enum", "enum_values": ["completed", "block_merge", "deploy_failed", "deploy_infra_failed", "pdc_failed", "unknown"],
                        "required": True,
                        "routing": [
                            {"predicate": {"operator": "equals", "value": "completed"}, "outcome": "success"},
                            {"predicate": {"operator": "equals", "value": "block_merge"}, "outcome": "failure"},
                            {"predicate": {"operator": "equals", "value": "deploy_failed"}, "outcome": "failure"},
                            {"predicate": {"operator": "equals", "value": "deploy_infra_failed"}, "outcome": "error"},
                            {"predicate": {"operator": "equals", "value": "pdc_failed"}, "outcome": "failure"},
                            {"predicate": {"operator": "equals", "value": "unknown"}, "outcome": "error"},
                        ],
                    },
                    {"canonical_path": "evidence.merge.merge_commit", "label": "Merge commit", "description": "Commit produced by the merge, when available.", "value_type": "string", "required": False, "routing": []},
                    {"canonical_path": "evidence.merge.final_stage", "label": "Final shipping stage", "description": "Last stage reached by the deterministic shipping pipeline.", "value_type": "string", "required": True, "routing": []},
                    {"canonical_path": "evidence.merge.governance_exceptions", "label": "Governance exceptions", "description": "Controls that could not be re-evaluated because the pull request was already merged.", "value_type": "array", "constraints": {"items": {"type": "string"}}, "required": False, "routing": []},
                ],
            },
            default_timeout_seconds=600,
            mutating=True,
            risk_level="high",
            requires_workspace=True,
            source_path="builtin://operations/ship_pipeline_v1",
            compatible_contract_versions=(1,),
        ),
        ExecutorManifestDescriptor(
            id="skip_validation_v1",
            executor_contract_version=1,
            domain_id="core",
            category="Control",
            kind="action",
            supported_phases=("main",),
            parameter_schema={
                "type": "object",
                "properties": {"reason": {"type": "string"}},
            },
            output_schema={
                "type": "object",
                "properties": {
                    "skip_validation_outcome": {"type": "string"},
                    "skip_validation_reason": {"type": "string"},
                    "node_outcome": {"type": "string"},
                },
            },
            default_timeout_seconds=600,
            mutating=False,
            risk_level="low",
            requires_workspace=False,
            source_path="builtin://operations/skip_validation_v1",
        ),
        ExecutorManifestDescriptor(
            id="ticket.has_acceptance_criteria",
            executor_contract_version=1,
            domain_id="core",
            category="Ticket context",
            kind="check",
            supported_phases=("preflight",),
            parameter_schema={"type": "object", "properties": {}},
            output_schema={
                "type": "object",
                "properties": {"count": {"type": "integer"}},
            },
            default_timeout_seconds=120,
            mutating=False,
            risk_level="low",
            requires_workspace=False,
            source_path="builtin://executors/ticket.has_acceptance_criteria",
            requires={"connections": [], "tools": []},
        ),
        ExecutorManifestDescriptor(
            id="ticket.has_branch",
            executor_contract_version=1,
            domain_id="core",
            category="Ticket context",
            kind="check",
            supported_phases=("preflight",),
            parameter_schema={"type": "object", "properties": {}},
            output_schema={
                "type": "object",
                "properties": {
                    "field": {"type": "string"},
                    "branch": {"type": "string"},
                },
            },
            default_timeout_seconds=120,
            mutating=False,
            risk_level="low",
            requires_workspace=False,
            source_path="builtin://executors/ticket.has_branch",
            requires={"connections": [], "tools": []},
        ),
        ExecutorManifestDescriptor(
            id="ticket.has_description",
            executor_contract_version=1,
            domain_id="core",
            category="Ticket context",
            kind="check",
            supported_phases=("preflight",),
            parameter_schema={"type": "object", "properties": {}},
            output_schema={
                "type": "object",
                "properties": {"length": {"type": "integer"}},
            },
            default_timeout_seconds=120,
            mutating=False,
            risk_level="low",
            requires_workspace=False,
            source_path="builtin://executors/ticket.has_description",
            requires={"connections": [], "tools": []},
        ),
        ExecutorManifestDescriptor(
            id="ticket.has_priority",
            executor_contract_version=1,
            domain_id="core",
            category="Ticket context",
            kind="check",
            supported_phases=("preflight",),
            parameter_schema={"type": "object", "properties": {}},
            output_schema={
                "type": "object",
                "properties": {"priority": {"type": "string"}},
            },
            default_timeout_seconds=120,
            mutating=False,
            risk_level="low",
            requires_workspace=False,
            source_path="builtin://executors/ticket.has_priority",
            requires={"connections": [], "tools": []},
        ),
        ExecutorManifestDescriptor(
            id="ticket.has_repo",
            executor_contract_version=1,
            domain_id="core",
            category="Ticket context",
            kind="check",
            supported_phases=("preflight",),
            parameter_schema={"type": "object", "properties": {}},
            output_schema={
                "type": "object",
                "properties": {
                    "repo": {"type": "string"},
                    "repo_url": {"type": "string"},
                },
            },
            default_timeout_seconds=120,
            mutating=False,
            risk_level="low",
            requires_workspace=False,
            source_path="builtin://executors/ticket.has_repo",
            requires={"connections": [], "tools": []},
        ),
        ExecutorManifestDescriptor(
            id="ticket.has_workflow",
            executor_contract_version=1,
            domain_id="core",
            category="Ticket context",
            kind="check",
            supported_phases=("preflight",),
            parameter_schema={"type": "object", "properties": {}},
            output_schema={
                "type": "object",
                "properties": {
                    "workflow_template_id": {"type": "string"},
                    "workflow_node_id": {"type": "string"},
                },
            },
            default_timeout_seconds=120,
            mutating=False,
            risk_level="low",
            requires_workspace=False,
            source_path="builtin://executors/ticket.has_workflow",
            requires={"connections": [], "tools": []},
        ),
        ExecutorManifestDescriptor(
            id="workspace.exists",
            executor_contract_version=1,
            domain_id="core",
            category="Repository",
            kind="check",
            supported_phases=("preflight",),
            parameter_schema={"type": "object", "properties": {}},
            output_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
            default_timeout_seconds=120,
            mutating=False,
            risk_level="low",
            requires_workspace=True,
            source_path="builtin://executors/workspace.exists",
            requires={"connections": [], "tools": []},
        ),
    )
)
