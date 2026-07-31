"""Offboarding retention policy.

When you leave an employer you need to separate two things that are usually
tangled together in a Downloads folder:

* records that are **yours** — immigration paperwork, payslips, tax forms,
  offer and relieving letters, certifications, personal projects;
* the employer's **work product** and its customers' data, which stays.

``RetentionPolicy`` is a thin layer over the existing category taxonomy. It
does not classify anything itself — it maps an already-assigned category to a
``Disposition``. Keeping it separate from ``categories.yaml`` means the same
classifier output can be re-interpreted under a different policy (a different
employer, a different jurisdiction) without touching classification.

Two safety properties are deliberate:

1. **Fail closed.** A category absent from the policy raises rather than
   defaulting, so adding a category later cannot silently land in ``keep``.
2. **Independent export guard.** ``never_export`` is checked separately from
   the disposition, so a typo that lists a secret-bearing category under
   ``keep`` still cannot write it into the bundle.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable

import yaml


class Disposition(str, Enum):
    KEEP = "keep"
    LEAVE = "leave"
    REVIEW = "review"


class PolicyError(ValueError):
    """Raised when the policy and the category set disagree."""


@dataclass
class ExportResult:
    exported: list[tuple[Path, Path]] = field(default_factory=list)
    skipped_never_export: list[Path] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.exported)


@dataclass
class RetentionPolicy:
    by_category: dict[str, Disposition]
    never_export: frozenset[str]

    # ------------------------------------------------------------- loading

    @classmethod
    def load(cls, path: str | Path) -> "RetentionPolicy":
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        dispositions = raw.get("dispositions") or {}

        by_category: dict[str, Disposition] = {}
        for name in ("keep", "leave", "review"):
            for category in dispositions.get(name) or []:
                if category in by_category:
                    raise PolicyError(
                        f"category {category!r} appears in more than one disposition list"
                    )
                by_category[category] = Disposition(name)

        return cls(
            by_category=by_category,
            never_export=frozenset(raw.get("never_export") or []),
        )

    # ------------------------------------------------------------ querying

    def validate(self, categories: Iterable[str]) -> None:
        """Fail loudly if the policy and the taxonomy have drifted apart."""
        known = set(categories)
        missing = sorted(known - set(self.by_category))
        unknown = sorted(set(self.by_category) - known)
        problems = []
        if missing:
            problems.append(
                "categories with no disposition (add them to config/offboarding.yaml): "
                + ", ".join(missing)
            )
        if unknown:
            problems.append(
                "policy names categories that no longer exist: " + ", ".join(unknown)
            )
        if problems:
            raise PolicyError("; ".join(problems))

    def disposition(self, category: str) -> Disposition:
        try:
            return self.by_category[category]
        except KeyError:
            raise PolicyError(
                f"category {category!r} has no disposition in the offboarding policy"
            ) from None

    def is_exportable(self, category: str) -> bool:
        """Exportable means KEEP *and* not independently barred."""
        return (
            self.disposition(category) is Disposition.KEEP
            and category not in self.never_export
        )

    def partition(self, plan: dict[str, object]) -> dict[Disposition, list[tuple[str, str]]]:
        """Split a classification plan into (path, category) lists by disposition."""
        out: dict[Disposition, list[tuple[str, str]]] = {d: [] for d in Disposition}
        for filepath, classification in plan.items():
            category = getattr(classification, "category", None)
            if category is None and isinstance(classification, dict):
                category = classification.get("category")
            out[self.disposition(str(category))].append((filepath, str(category)))
        return out

    # ------------------------------------------------------------ exporting

    def export(
        self,
        plan: dict[str, object],
        destination: Path,
        apply: bool = False,
        move: bool = False,
    ) -> ExportResult:
        """Copy (default) or move KEEP files into ``destination/<Category>/``.

        Copy is the default: originals stay put until you have verified the
        bundle. Nothing is written unless ``apply`` is True.
        """
        result = ExportResult()
        for filepath, classification in plan.items():
            category = getattr(classification, "category", None)
            if category is None and isinstance(classification, dict):
                category = classification.get("category")
            category = str(category)

            if self.disposition(category) is not Disposition.KEEP:
                continue
            if category in self.never_export:
                result.skipped_never_export.append(Path(filepath))
                continue

            src = Path(filepath)
            dest_dir = destination / category
            dest = dest_dir / src.name

            counter = 1
            while dest.exists() or any(d == dest for _, d in result.exported):
                dest = dest_dir / f"{src.stem}_{counter}{src.suffix}"
                counter += 1

            if apply:
                try:
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    if move:
                        shutil.move(str(src), str(dest))
                    else:
                        shutil.copy2(str(src), str(dest))
                except OSError as e:
                    result.errors.append(f"{src} -> {dest}: {e}")
                    continue
            result.exported.append((src, dest))
        return result
