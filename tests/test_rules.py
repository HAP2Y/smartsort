"""Filename-classification regression tests for the SmartSort RulesEngine.

Each case is a real (or representative) filename that the project has seen,
paired with the category it should be classified into. These tests guard the
fixes for the underscore / camelCase word-boundary bug and the EVL/PR routing.
"""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from classifier.rules import RulesEngine  # noqa: E402

CONFIG = str(ROOT / "config" / "categories.yaml")


@pytest.fixture(scope="module")
def rules():
    return RulesEngine(CONFIG)


def _classify(rules, name):
    hc = rules.high_confidence_match(name)
    if hc:
        return hc.category, hc.confidence, hc.reason, "HC"
    c = rules.classify(name)
    return c.category, c.confidence, c.reason, "Rules"


CANADIAN_PR_CASES = [
    "Happy_imm5476e_Signed.pdf",
    "Happy_imm5476e.pdf",
    "IMM5787_2-1I6353LE.pdf",
    "IMM5790_2-1HWLQ6BD.pdf",
    "IMM10003_2-1I6353LK.pdf",
    "imm5644e.pdf",
    "Patel_Happy_2022-2023_T4.pdf",
    "Patel_Happy_2022-2023_T4_unlocked.pdf",
    "previewFormPCCDetail.pdf",
    "MSUBARODA_WESEducationalCredentialsForwarding.pdf",
    "Happy Pravinbhai Patel POST ITA 1250.pdf",
    "PR_GWRE_Experience_Letter_Info_-_Happy_Patel.pdf",
    "Canada_Happy Patel_Employment Verification Letter.pdf",
    "Kunal Khosla Employment verification letter.docx",
    "Kunal Khosla Employment verification letter.pdf",
    "Employment Verification Letter - with comp details_HappyPatel.pdf",
    "employment_verification_letter_-_with_c.pdf",
]

FINANCIAL_CASES = [
    "PATEL_CAN_PAY_SLIP_JUN_2025.pdf",
    "Patel_Happy_HDFC_BalanceCertificate_269.pdf",
    "CH_BalanceCertificate_269076050.pdf",
    "Mobile Receipt.pdf",
    "April-2026_HDFC_Credit_Card.pdf",
    "March-2026_Jio_Home.pdf",
]

MEDICAL_CASES = [
    "Happy_Doctors_Follow_up_Letter.pdf",
    "Excellence_Ulticaria_to_Heat_Hives_Note.pdf",
    "Medical Documentation Submission - Happy Patel.eml",
]

FRANCHISE_CASES = [
    "Reality_Flip_Complete.html",
    "Reality Flip.html",
    "Reality Flip.zip",
    "franchise_research_framework.docx",
]

ASTRO_CASES = ["xlb.csv", "xlc.csv", "xlk.csv", "xlre.csv", "xlv.csv", "nfty.csv"]

ARCHIVE_CASES = [
    "Ollama.dmg",
    "Claude.dmg",
    "Untitled design.zip",
    "job_hunter.zip",
    "Sectigo_Root_cert.zip",
]

METADATA_CASES = [
    ".DS_Store",
    ".localized",
    "Sectigo Public Server Authentication Root R46.cer",
]

RESUME_CASES = [
    "HAPPY PATEL - Resume - EllevenLabs.pdf",
    "Happy Patel Resume.pdf",
    "CV Building with AI Skills - Google Gemini.pdf",
]

MEDIA_CASES = [
    "IMG-20260125-WA0002.jpg",
    "image.png",
    "ChatGPT Image Apr 28, 2026, 08_14_52 PM.png",
]

TRAVEL_CASES = [
    "IndiGo Itinerary - Y4947Y.pdf",
    "Air India Web Booking eTicket (JGF3FY) - HAPPY PRAVINBHAI.pdf",
    "OCTAVE-20thApril.pdf",
]


@pytest.mark.parametrize("name", CANADIAN_PR_CASES)
def test_canadian_pr(rules, name):
    cat, *_ = _classify(rules, name)
    assert cat == "Canadian_PR_Docs", name


@pytest.mark.parametrize("name", FINANCIAL_CASES)
def test_financial(rules, name):
    cat, *_ = _classify(rules, name)
    assert cat == "Financial_Taxes", name


@pytest.mark.parametrize("name", MEDICAL_CASES)
def test_medical(rules, name):
    cat, *_ = _classify(rules, name)
    assert cat == "Medical_Health", name


@pytest.mark.parametrize("name", FRANCHISE_CASES)
def test_franchise(rules, name):
    cat, *_ = _classify(rules, name)
    assert cat == "Franchise_Business_Research", name


@pytest.mark.parametrize("name", ASTRO_CASES)
def test_astro(rules, name):
    cat, *_ = _classify(rules, name)
    assert cat == "AstroQuant_Sidereal", name


@pytest.mark.parametrize("name", ARCHIVE_CASES)
def test_archives(rules, name):
    cat, *_ = _classify(rules, name)
    assert cat == "Archives_and_Apps", name


@pytest.mark.parametrize("name", METADATA_CASES)
def test_metadata(rules, name):
    cat, *_ = _classify(rules, name)
    assert cat == "Metadata_System", name


@pytest.mark.parametrize("name", RESUME_CASES)
def test_resume(rules, name):
    cat, *_ = _classify(rules, name)
    assert cat == "Resumes_Career_Tech", name


@pytest.mark.parametrize("name", MEDIA_CASES)
def test_media(rules, name):
    cat, *_ = _classify(rules, name)
    assert cat == "Media_Images", name


@pytest.mark.parametrize("name", TRAVEL_CASES)
def test_travel(rules, name):
    cat, *_ = _classify(rules, name)
    assert cat == "Travel_Transit", name


def test_print_pdf_does_not_match_pr(rules):
    """Print.pdf must not be tokenised into a 'pr' match."""
    cat, *_ = _classify(rules, "Print.pdf")
    assert cat != "Canadian_PR_Docs"


def test_tokenizer_handles_camelcase_and_underscores(rules):
    tokens, flat = RulesEngine._tokenize("previewFormPCCDetail.pdf")
    assert "pcc" in tokens
    assert "preview" in tokens
    assert "detail" in tokens
    assert "pcc" in flat
