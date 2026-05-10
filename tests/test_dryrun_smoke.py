"""End-to-end smoke test: spawn `main.py run <tmp> --no-ai` against a fixture
directory of representative empty files and assert the printed classification
plan routes each file to the expected category.

This complements ``test_rules.py`` by exercising the full CLI pipeline
(rules engine + extractor + organizer wiring) in dry-run mode.
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FIXTURES = [
    ("Happy_imm5476e_Signed.pdf",                           "Canadian_PR_Docs"),
    ("Patel_Happy_2022-2023_T4.pdf",                        "Canadian_PR_Docs"),
    ("Canada_Happy Patel_Employment Verification Letter.pdf", "Canadian_PR_Docs"),
    ("PR_GWRE_Experience_Letter.pdf",                       "Canadian_PR_Docs"),
    ("PATEL_CAN_PAY_SLIP_JUN_2025.pdf",                     "Financial_Taxes"),
    ("Patel_Happy_HDFC_BalanceCertificate_269.pdf",         "Financial_Taxes"),
    ("Happy_Doctors_Follow_up_Letter.pdf",                  "Medical_Health"),
    ("xlre.csv",                                            "AstroQuant_Sidereal"),
    ("Reality_Flip_Complete.html",                          "Franchise_Business_Research"),
    ("Air India Web Booking eTicket - HAPPY.pdf",           "Travel_Transit"),
    ("HAPPY PATEL - Resume.pdf",                            "Resumes_Career_Tech"),
    ("IMG-20260125-WA0002.jpg",                             "Media_Images"),
    ("Ollama.dmg",                                          "Archives_and_Apps"),
]


def test_cli_dryrun_classifies_each_fixture(tmp_path: Path):
    for name, _ in FIXTURES:
        (tmp_path / name).touch()

    env = {**os.environ, "NO_COLOR": "1", "TERM": "dumb", "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run(
        [sys.executable, "main.py", "run", str(tmp_path), "--no-ai"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )

    assert result.returncode == 0, (
        f"CLI exited {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    output = result.stdout
    # The summary table prints each category name verbatim.
    for _, expected_cat in FIXTURES:
        assert expected_cat in output, (
            f"Expected category {expected_cat!r} missing from CLI output:\n{output}"
        )

    # Dry-run must NOT have moved any files.
    for name, _ in FIXTURES:
        assert (tmp_path / name).exists(), f"{name} should still exist after dry-run"

    # Dry-run must NOT have written an undo log.
    assert not (tmp_path / ".smartsort_undo.json").exists()
