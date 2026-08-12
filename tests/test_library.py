from __future__ import annotations

import json
from pathlib import Path

from sickr_workflow_executors.executor_manifest import (
    BUILTIN_EXECUTOR_MANIFESTS,
    assert_registration_conformance,
)
from sickr_workflow_executors.workflow_executor_steps import DEFAULT_STEP_EXECUTORS


def test_every_handler_has_exactly_one_manifest() -> None:
    step_descriptors = {
        executor_id: descriptor
        for executor_id, descriptor in BUILTIN_EXECUTOR_MANIFESTS.items()
        if any(phase in descriptor.supported_phases for phase in ("preflight", "postflight"))
    }
    assert_registration_conformance(
        descriptors=step_descriptors,
        registrations=DEFAULT_STEP_EXECUTORS,
        registration_name="SICKR executor library",
    )


def test_committed_manifest_matches_runtime_registry() -> None:
    fixture = json.loads((Path(__file__).parents[1] / "manifests" / "builtin-executors.v1.json").read_text(encoding="utf-8"))
    assert {item["id"] for item in fixture["executors"]} == set(BUILTIN_EXECUTOR_MANIFESTS)
