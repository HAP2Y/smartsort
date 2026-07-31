"""Tests for the offboarding retention policy.

The safety-critical properties, in order of how much damage a regression
would do:

1. Credential-bearing files are never exported — not under ``--apply``, not
   under ``--move``, not even if a policy edit mistakenly lists them as KEEP.
2. The employee's own HR paperwork is KEEP even when the employer's name is
   in the filename. A real fixture regressed here:
   "Guidewire Mail - Resignation - Happy Patel.pdf" classified as company
   work product and would have been left behind.
3. Company work product and customer data are never exported.
4. The policy fails closed when it drifts from the category taxonomy.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from classifier.classifiers import HighConfidenceRulesClassifier, RulesClassifier
from classifier.pipeline import ClassificationPipeline
from classifier.rules import RulesEngine
from classifier.types import Classification, FileItem
from movers.offboarding import Disposition, PolicyError, RetentionPolicy

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = str(ROOT / "config" / "categories.yaml")
POLICY = str(ROOT / "config" / "offboarding.yaml")


@pytest.fixture(scope="module")
def policy():
    return RetentionPolicy.load(POLICY)


@pytest.fixture(scope="module")
def classify():
    rules = RulesEngine(CATEGORIES)
    pipe = ClassificationPipeline([
        HighConfidenceRulesClassifier(rules), RulesClassifier(rules),
    ])
    return lambda p: pipe.classify(FileItem(path=p))


def _plan(tmp_path, names, classify):
    plan = {}
    for n in names:
        p = tmp_path / n
        p.write_text("x")
        plan[str(p)] = classify(p)
    return plan


# ------------------------------------------------------------------ policy


def test_policy_covers_every_category(policy):
    """Fail-closed guard: a new category must be given a disposition."""
    cats = list(yaml.safe_load(open(CATEGORIES))["categories"].keys())
    policy.validate(cats)  # must not raise


def test_policy_rejects_missing_category(policy):
    with pytest.raises(PolicyError, match="no disposition"):
        policy.validate(["Canadian_PR_Docs", "A_Brand_New_Category"])


def test_policy_rejects_duplicate_disposition(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({
        "dispositions": {"keep": ["X"], "leave": ["X"]},
    }))
    with pytest.raises(PolicyError, match="more than one disposition"):
        RetentionPolicy.load(bad)


def test_unknown_category_raises_rather_than_defaulting(policy):
    with pytest.raises(PolicyError):
        policy.disposition("Not_A_Category")


# ------------------------------------------------------- credential safety


@pytest.mark.parametrize("name", [
    "github-recovery-codes.txt",
    "tenant-ups-db-xfer_accessKeys.csv",
    "id_rsa",
])
def test_credentials_are_review_and_never_exportable(policy, classify, tmp_path, name):
    p = tmp_path / name
    p.write_text("x")
    cat = classify(p).category
    assert cat == "Credentials_Secrets", f"{name} -> {cat}"
    assert policy.disposition(cat) is Disposition.REVIEW
    assert not policy.is_exportable(cat)


def test_credentials_never_written_even_if_policy_says_keep(tmp_path, classify):
    """The never_export guard is independent of the disposition list."""
    rogue = tmp_path / "rogue.yaml"
    rogue.write_text(yaml.safe_dump({
        "dispositions": {"keep": ["Credentials_Secrets"], "leave": [], "review": []},
        "never_export": ["Credentials_Secrets"],
    }))
    pol = RetentionPolicy.load(rogue)
    assert pol.disposition("Credentials_Secrets") is Disposition.KEEP  # misconfigured
    assert not pol.is_exportable("Credentials_Secrets")               # still barred

    plan = _plan(tmp_path, ["github-recovery-codes.txt"], classify)
    dest = tmp_path / "bundle"
    result = pol.export(plan, dest, apply=True)
    assert result.count == 0
    assert len(result.skipped_never_export) == 1
    assert not dest.exists() or not any(dest.rglob("*recovery*"))


def test_move_flag_does_not_bypass_never_export(tmp_path, classify, policy):
    plan = _plan(tmp_path, ["tenant-ups-db-xfer_accessKeys.csv"], classify)
    src = Path(next(iter(plan)))
    result = policy.export(plan, tmp_path / "bundle", apply=True, move=True)
    assert result.count == 0
    assert src.exists(), "a never-export file must not be moved away either"


# --------------------------------------------------------- HR vs work product


@pytest.mark.parametrize("name", [
    # Employer name in the filename must not make these company property.
    "Guidewire Mail - Resignation - Happy Patel - Platform Support Engineer II.pdf",
    "IN - Offer of Employment_Happy Patel (208983).pdf",
    "International Transfer Letter_Happy Patel (208983).pdf",
    "Patel, Happy Flex Work Letter.pdf",
])
def test_own_hr_paperwork_is_kept(policy, classify, tmp_path, name):
    p = tmp_path / name
    p.write_text("x")
    cat = classify(p).category
    assert cat == "Employment_Records", f"{name} -> {cat}"
    assert policy.disposition(cat) is Disposition.KEEP


@pytest.mark.parametrize("name,expected", [
    ("InsuranceSuite - Platform Specifications and Limits - Production.pdf", "Guidewire_PSE_Work"),
    ("IS.02 Enterprise Acceptable Use Policy.pdf", "Guidewire_PSE_Work"),
    ("DigitalPortals_Test_2469.log", "Guidewire_PSE_Work"),
    ("CPIKOR-CORP-Customer-metadata.xml", "Customer_Data"),
    ("ZURICHCI-PROD-Customer-metadata.xml", "Customer_Data"),
])
def test_company_and_customer_material_is_left(policy, classify, tmp_path, name, expected):
    p = tmp_path / name
    p.write_text("x")
    cat = classify(p).category
    assert cat == expected, f"{name} -> {cat}"
    assert policy.disposition(cat) is Disposition.LEAVE
    assert not policy.is_exportable(cat)


@pytest.mark.parametrize("name", [
    "Happy_imm5476e_Signed.pdf",
    "Patel_Happy_2022-2023_T4.pdf",
    "Happy_Tax_Forms_2025.pdf",
    "proof_of_income_2025_en.pdf",
    "PATEL_CAN_PAY_SLIP_JUN_2025.pdf",
    "Payslip_May_2026.pdf",
    "HAPPY PATEL - Resume - SumoLogic.pdf",
])
def test_personal_records_are_kept(policy, classify, tmp_path, name):
    p = tmp_path / name
    p.write_text("x")
    cat = classify(p).category
    assert policy.disposition(cat) is Disposition.KEEP, f"{name} -> {cat}"


# ------------------------------------------------------------------ export


def test_export_is_dry_run_by_default(tmp_path, classify, policy):
    plan = _plan(tmp_path, ["Happy_imm5476e_Signed.pdf"], classify)
    dest = tmp_path / "bundle"
    result = policy.export(plan, dest, apply=False)
    assert result.count == 1          # reported
    assert not dest.exists()          # but nothing written


def test_export_copies_and_leaves_originals(tmp_path, classify, policy):
    plan = _plan(tmp_path, ["Happy_imm5476e_Signed.pdf"], classify)
    src = Path(next(iter(plan)))
    dest = tmp_path / "bundle"
    policy.export(plan, dest, apply=True)
    assert src.exists(), "copy is the default so originals survive"
    assert (dest / "Canadian_PR_Docs" / src.name).exists()


def test_export_move_removes_original(tmp_path, classify, policy):
    plan = _plan(tmp_path, ["Happy_imm5476e_Signed.pdf"], classify)
    src = Path(next(iter(plan)))
    dest = tmp_path / "bundle"
    policy.export(plan, dest, apply=True, move=True)
    assert not src.exists()
    assert (dest / "Canadian_PR_Docs" / src.name).exists()


def test_export_excludes_leave_and_review(tmp_path, classify, policy):
    plan = _plan(tmp_path, [
        "Happy_imm5476e_Signed.pdf",                    # keep
        "CPIKOR-CORP-Customer-metadata.xml",            # leave
        "github-recovery-codes.txt",                    # review / never
    ], classify)
    dest = tmp_path / "bundle"
    result = policy.export(plan, dest, apply=True)
    assert result.count == 1
    written = {p.name for p in dest.rglob("*") if p.is_file()}
    assert written == {"Happy_imm5476e_Signed.pdf"}


def test_partition_totals_match_input(tmp_path, classify, policy):
    names = [
        "Happy_imm5476e_Signed.pdf",
        "IN - Offer of Employment_Happy Patel.pdf",
        "CPIKOR-CORP-Customer-metadata.xml",
        "github-recovery-codes.txt",
        "Print.pdf",
    ]
    plan = _plan(tmp_path, names, classify)
    buckets = policy.partition(plan)
    assert sum(len(v) for v in buckets.values()) == len(names)


def test_partition_accepts_dict_classifications(tmp_path, policy):
    """Plans round-tripped through the queue arrive as dicts, not dataclasses."""
    plan = {"/x/a.pdf": {"category": "Canadian_PR_Docs", "confidence": 100}}
    buckets = policy.partition(plan)
    assert buckets[Disposition.KEEP] == [("/x/a.pdf", "Canadian_PR_Docs")]


def test_export_suffixes_name_collisions(tmp_path, classify, policy):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir(); b.mkdir()
    plan = {}
    for d in (a, b):
        p = d / "Happy_imm5476e_Signed.pdf"
        p.write_text("x")
        plan[str(p)] = classify(p)
    dest = tmp_path / "bundle"
    result = policy.export(plan, dest, apply=True)
    assert result.count == 2
    names = sorted(p.name for p in (dest / "Canadian_PR_Docs").iterdir())
    assert names == ["Happy_imm5476e_Signed.pdf", "Happy_imm5476e_Signed_1.pdf"]
