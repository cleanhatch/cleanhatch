"""GOV-01/GOV-02 contribution and licensing entry-point checks."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRIBUTING = PROJECT_ROOT / "CONTRIBUTING.md"
PR_TEMPLATE = PROJECT_ROOT / ".github" / "PULL_REQUEST_TEMPLATE.md"
ISSUE_DIR = PROJECT_ROOT / ".github" / "ISSUE_TEMPLATE"
BUG_FORM = ISSUE_DIR / "bug_report.yml"
PITFALL_FORM = ISSUE_DIR / "silent_killer_pitfall.yml"
ISSUE_CONFIG = ISSUE_DIR / "config.yml"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(_read(path))


def _plain(text: str) -> str:
    """Collapse wrapping and Markdown emphasis for readable assertions."""
    return " ".join(text.replace("*", "").replace("`", "").lower().split())


def _by_id(fields: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {field["id"]: field for field in fields if "id" in field}


def _checkbox_labels(fields: list[dict[str, Any]]) -> list[str]:
    return [
        option["label"]
        for field in fields
        if field["type"] == "checkboxes"
        for option in field["attributes"]["options"]
    ]


@pytest.mark.parametrize("path", [BUG_FORM, PITFALL_FORM], ids=["bug", "pitfall"])
def test_issue_form_schema_and_required_fields(path: Path) -> None:
    form = _load(path)
    assert isinstance(form["name"], str) and form["name"]
    assert isinstance(form["description"], str) and form["description"]
    assert len(form["description"]) <= 200
    assert isinstance(form["title"], str) and form["title"]

    body = form["body"]
    assert isinstance(body, list) and body
    assert any(field["type"] != "markdown" for field in body)
    allowed = {"markdown", "input", "textarea", "dropdown", "checkboxes"}
    ids: list[str] = []
    labels: list[str] = []
    for field in body:
        assert field["type"] in allowed
        if field["type"] == "markdown":
            assert field["attributes"]["value"].strip()
            assert "id" not in field
            continue
        field_id = field["id"]
        label = field["attributes"]["label"]
        assert re.fullmatch(r"[A-Za-z0-9_-]+", field_id)
        assert isinstance(label, str) and label.strip()
        ids.append(field_id)
        labels.append(label)
        if field["type"] == "checkboxes":
            options = field["attributes"]["options"]
            assert isinstance(options, list) and options
            for option in options:
                assert isinstance(option["label"], str) and option["label"].strip()
                assert isinstance(option["required"], bool)
        else:
            assert isinstance(field["validations"]["required"], bool)
            if field["type"] == "dropdown":
                options = field["attributes"]["options"]
                assert isinstance(options, list) and options
                assert all(isinstance(option, str) and option.strip() for option in options)
                assert len(options) == len(set(options))
            else:
                assert "options" not in field["attributes"]
    assert len(ids) == len(set(ids))
    assert len(labels) == len(set(labels))
    checkbox_labels = _checkbox_labels(body)
    assert len(checkbox_labels) == len(set(checkbox_labels))


def test_bug_form_is_a_bug_entry_not_a_license_grant() -> None:
    fields = _load(BUG_FORM)["body"]
    by_id = _by_id(fields)
    for required_id in ("bug-summary", "environment", "reproduction", "expected-actual"):
        assert by_id[required_id]["validations"]["required"] is True
    assert by_id["logs"]["validations"]["required"] is False
    assert "attribution" not in by_id
    assert "contribution-license" not in by_id
    assert "contribution-tier" not in by_id
    hygiene = by_id["data-hygiene"]["attributes"]["options"]
    assert hygiene and all(option["required"] is True for option in hygiene)

    labels = " ".join(_checkbox_labels(fields)).lower()
    for forbidden in (
        "grant",
        "copyright",
        "patent",
        "cla",
        "dco",
        "perpetual",
        "irrevocable",
        "moral",
    ):
        assert forbidden not in labels

    text = _plain(_read(BUG_FORM))
    assert "do not need to buy l1" in text
    assert "not a contribution" in text
    assert "apache-2.0" in text


def test_pitfall_form_has_explicit_tier1_apache_acceptance() -> None:
    fields = _load(PITFALL_FORM)["body"]
    by_id = _by_id(fields)
    for required_id in (
        "attribution",
        "contribution-tier",
        "domain-pack",
        "symptom",
        "schema-and-time",
        "reproduction",
    ):
        assert by_id[required_id]["validations"]["required"] is True
    assert by_id["impact"]["validations"]["required"] is False
    for checkbox_id in ("data-hygiene", "contribution-license"):
        options = by_id[checkbox_id]["attributes"]["options"]
        assert options and all(option["required"] is True for option in options)

    license_labels = [
        option["label"] for option in by_id["contribution-license"]["attributes"]["options"]
    ]
    assert any("Apache License 2.0" in label for label in license_labels)
    assert any("beyond Apache-2.0" in label for label in license_labels)
    assert any("separate, specific written agreement" in label for label in license_labels)

    attribution = by_id["attribution"]["attributes"]
    assert "Anonymous" in attribution["label"]
    assert "Anonymous" in attribution["description"]

    text = _plain(_read(PITFALL_FORM))
    assert "tier 2" in text
    assert "pr template" in text
    assert "github still records the issue author" in text
    assert "no extra-rights grant is requested here" in text
    assert "do not need to buy l1" in text
    assert "contributing.md" in text
    assert "pull_request_template.md" in text

    labels = " ".join(_checkbox_labels(fields)).lower()
    for forbidden in ("perpetual", "irrevocable", "moral", "patent", "copyright", "waive"):
        assert forbidden not in labels


def test_pr_template_links_license_and_does_not_grant_extra_rights() -> None:
    text = _read(PR_TEMPLATE)
    plain = _plain(text)
    for marker in (
        "../contributing.md",
        "../license",
        "apache-2.0",
        "no cla is required",
        "no l1 purchase is required",
        "dco: not required today",
        "not a cla",
        "does not grant rights beyond apache-2.0",
        "separate written agreement",
    ):
        assert marker in plain
    assert "you may accept or decline" in plain
    assert "declining does not block" in plain

    licensing = text.split("## Licensing", 1)[1].split("## Checklist", 1)[0].lower()
    assert "- [ ]" not in licensing
    checkbox_lines = [
        line.strip().lower() for line in text.splitlines() if line.strip().startswith("- [ ]")
    ]
    assert checkbox_lines
    for line in checkbox_lines:
        for forbidden in (
            "additional rights",
            "extra rights",
            "relicense",
            "i grant",
            "agree to grant",
        ):
            assert forbidden not in line


def test_contributing_states_apache_commercial_use_and_dco_boundary() -> None:
    text = _plain(_read(CONTRIBUTING))
    for marker in (
        "no cla is required",
        "apache-2.0 already permits commercial use",
        "not an extra cla",
        "notice",
        "you keep your copyright",
        "we do not ask for a standing cla",
        "blanket waiver of moral rights",
        "does not currently require a developer certificate of origin",
        "not a merge gate today",
        "not a cla",
        "provenance",
        "anonymous",
        "cannot make your github account anonymous",
        "do not need to buy l1",
    ):
        assert marker in text
    assert "otherwise cannot enter l1" not in text
    assert "must purchase" not in text


def test_tier1_and_tier2_are_separate_entry_points() -> None:
    contributing = _read(CONTRIBUTING)
    pitfall = _load(PITFALL_FORM)
    bug = _load(BUG_FORM)
    pr = _read(PR_TEMPLATE)
    assert "## Tier 1" in contributing
    assert "## Tier 2" in contributing
    assert "scenario text and synthetic examples" in contributing.lower()
    assert "code contributions" in contributing.lower()
    pitfall_by_id = _by_id(pitfall["body"])
    assert "contribution-tier" in pitfall_by_id
    assert "contribution-license" in pitfall_by_id
    options = pitfall_by_id["contribution-tier"]["attributes"]["options"]
    assert any(option.startswith("Tier 1") for option in options)
    assert any(option.startswith("Tier 2") for option in options)
    assert "Tier 1" in pr and "Tier 2" in pr
    bug_ids = set(_by_id(bug["body"]))
    assert "contribution-tier" not in bug_ids
    assert "attribution" not in bug_ids
    assert "contribution-license" not in bug_ids


def test_declining_extra_rights_does_not_block_apache_review() -> None:
    text = _plain(_read(CONTRIBUTING))
    assert "declining is not by itself a reason to reject the contribution" in text
    assert "still review it, attribute it, and consider it for rewards under apache-2.0" in text
    assert "mergeability is decided separately on objective project reasons" in text
    assert "technical review requirements are not conditions for extra rights" in text
    assert "general review right is independent of any extra-rights decision" in text


def test_extra_rights_require_specific_recorded_consent() -> None:
    text = _plain(_read(CONTRIBUTING))
    for marker in (
        "exact additional rights",
        "exact agreement text",
        "version",
        "recorded affirmative action",
        "only the identified contribution",
        "what is not consent",
        "buying l1",
        "willing to be contacted",
        "silence",
        "draft purchase document",
    ):
        assert marker in text


def test_issue_config_keeps_security_and_adds_contribution_navigation() -> None:
    config = _load(ISSUE_CONFIG)
    assert config["blank_issues_enabled"] is False
    links = {link["name"]: link["url"] for link in config["contact_links"]}
    assert links["Contributing and licensing"].endswith("/CONTRIBUTING.md")
    assert links["Report a security issue privately"].endswith("/security/advisories/new")
    urls = list(links.values())
    assert any("cleanhatch.dev/#pricing" in url for url in urls)
    assert any("cleanhatch.dev/refund/" in url for url in urls)
    assert any("docs.cleanhatch.dev" in url for url in urls)
