from __future__ import annotations

from pathlib import Path

from sickr_workflow_executors.workflow_executor_steps import (
    _dependency_install_command,
    _nested_dependency_roots,
)


def test_repository_lockfile_selects_its_package_manager(tmp_path: Path) -> None:
    (tmp_path / "package-lock.json").write_text("{}", encoding="utf-8")
    assert _dependency_install_command(tmp_path) == "npm ci"


def test_discovers_one_nested_package_root(tmp_path: Path) -> None:
    worker = tmp_path / "worker"
    worker.mkdir()
    (worker / "package-lock.json").write_text("{}", encoding="utf-8")
    assert _nested_dependency_roots(tmp_path) == [(worker, "npm ci")]


def test_reports_multiple_independent_roots_deterministically(tmp_path: Path) -> None:
    api = tmp_path / "apps" / "api"
    ui = tmp_path / "apps" / "ui"
    api.mkdir(parents=True)
    ui.mkdir(parents=True)
    (api / "uv.lock").write_text("", encoding="utf-8")
    (ui / "pnpm-lock.yaml").write_text("", encoding="utf-8")
    assert _nested_dependency_roots(tmp_path) == [
        (api, "uv sync --frozen"),
        (ui, "pnpm install --frozen-lockfile"),
    ]


def test_ignores_dependency_and_hidden_directories(tmp_path: Path) -> None:
    ignored = [tmp_path / "node_modules" / "fixture", tmp_path / ".cache" / "fixture"]
    for directory in ignored:
        directory.mkdir(parents=True)
        (directory / "package-lock.json").write_text("{}", encoding="utf-8")
    assert _nested_dependency_roots(tmp_path) == []
