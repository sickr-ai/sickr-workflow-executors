from __future__ import annotations

import json
from pathlib import Path

from sickr_workflow_executors.executor_manifest import (
    BUILTIN_EXECUTOR_MANIFESTS,
    EXPECTED_BUILTIN_EXECUTOR_FIXTURE_SHA256,
    assert_registration_conformance,
    builtin_executor_fixture,
    stable_executor_fixture_sha256,
)
from sickr_workflow_executors.workflow_executor_steps import (
    DEFAULT_STEP_EXECUTORS,
    GitStepResult,
    _github_repository_access_failure,
)


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
    assert fixture == builtin_executor_fixture()
    assert stable_executor_fixture_sha256(fixture) == EXPECTED_BUILTIN_EXECUTOR_FIXTURE_SHA256


def test_github_repository_access_failure_is_actionable_only_for_access_errors() -> None:
    assert _github_repository_access_failure(
        GitStepResult(128, "", "remote: Repository not found.")
    )
    assert _github_repository_access_failure(
        GitStepResult(128, "", "fatal: Authentication failed for repository")
    )
    assert not _github_repository_access_failure(
        GitStepResult(128, "", "fatal: unable to access repository: connection timed out")
    )
