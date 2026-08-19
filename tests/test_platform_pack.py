import json
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_platform_pack_uses_fixed_folders_and_resolves_every_artifact():
    pack = json.loads((ROOT / "platform-pack.json").read_text(encoding="utf-8"))
    assert pack["schema_version"] == "sickr.platform_config_pack.v1"
    assert pack["stable"] is True
    folder_for_kind = {
        "executor_library": "executors",
        "skill": "skills",
        "skill_pack": "skills",
        "workflow": "workflows",
        "governance_state_type": "governance-states",
        "governance_rule": "governance-rules",
        "setting": "settings",
    }
    for artifact in pack["artifacts"]:
        path = Path(artifact["path"])
        assert path.parts[0] == folder_for_kind[artifact["kind"]]
        assert (ROOT / path).is_file()
        json.loads((ROOT / path).read_text(encoding="utf-8"))
