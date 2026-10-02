"""Append-only trial log. A result without a prior registration is refused."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

HOLDOUT_TRIAL = "holdout"
REPORT_ENTRY = "report"


class TrialLogError(RuntimeError):
    """The log-before-you-look rule would be broken."""


class RecordKind(Enum):
    REGISTERED = "registered"
    RESULT = "result"
    REPORT = "report"
    HOLDOUT_OPENED = "holdout_opened"
    HOLDOUT_RESULT = "holdout_result"


class TrialStatus(Enum):
    KEPT = "kept"
    DISCARDED = "discarded"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True)
class TrialLog:
    path: Path
    clock: Callable[[], str] = _utc_now

    def records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]

    def _ids(self, kind: RecordKind) -> set[str]:
        return {r["trial_id"] for r in self.records() if r["kind"] == kind.value}

    def _append(self, record: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # newline fixed so the file is byte-identical on every platform
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def is_registered(self, trial_id: str) -> bool:
        return trial_id in self._ids(RecordKind.REGISTERED)

    def has_result(self, trial_id: str) -> bool:
        return trial_id in self._ids(RecordKind.RESULT)

    def register(
        self,
        trial_id: str,
        hypothesis: str,
        config: Mapping[str, Any],
        snapshot_hash: str,
        code_commit: str,
    ) -> None:
        if self.is_registered(trial_id):
            raise TrialLogError(f"trial {trial_id} is already registered")
        self._append(
            {
                "kind": RecordKind.REGISTERED.value,
                "trial_id": trial_id,
                "at": self.clock(),
                "hypothesis": hypothesis,
                "config": dict(config),
                "snapshot_hash": snapshot_hash,
                "code_commit": code_commit,
            }
        )

    def record_result(
        self,
        trial_id: str,
        metrics: Mapping[str, Any],
        status: TrialStatus,
        reason: str,
        labels_path: str,
        labels_sha256: str,
    ) -> None:
        """`labels_sha256` ties the label file to this record: the report re-reads that file."""
        if not self.is_registered(trial_id):
            raise TrialLogError(f"trial {trial_id} has no prior registration")
        if self.has_result(trial_id):
            raise TrialLogError(f"trial {trial_id} already has a result")
        self._append(
            {
                "kind": RecordKind.RESULT.value,
                "trial_id": trial_id,
                "at": self.clock(),
                "metrics": dict(metrics),
                "status": status.value,
                "reason": reason,
                "labels_path": labels_path,
                "labels_sha256": labels_sha256,
            }
        )

    def results(self) -> dict[str, dict[str, Any]]:
        return {r["trial_id"]: r for r in self.records() if r["kind"] == RecordKind.RESULT.value}

    def registrations(self) -> dict[str, dict[str, Any]]:
        return {
            r["trial_id"]: r for r in self.records() if r["kind"] == RecordKind.REGISTERED.value
        }

    def holdout_opened(self) -> bool:
        return HOLDOUT_TRIAL in self._ids(RecordKind.HOLDOUT_OPENED)

    def open_holdout(self, final_trial_id: str) -> None:
        """Record, once and for ever, that the holdout has been looked at."""
        if self.holdout_opened():
            raise TrialLogError("the holdout has already been opened")
        if not self.has_result(final_trial_id):
            raise TrialLogError(f"final model {final_trial_id} has no recorded result")
        self._append(
            {
                "kind": RecordKind.HOLDOUT_OPENED.value,
                "trial_id": HOLDOUT_TRIAL,
                "at": self.clock(),
                "final_trial_id": final_trial_id,
            }
        )

    def record_holdout_result(self, payload: Mapping[str, Any]) -> None:
        """The one result of the one holdout evaluation."""
        if not self.holdout_opened():
            raise TrialLogError("the holdout has not been opened")
        if HOLDOUT_TRIAL in self._ids(RecordKind.HOLDOUT_RESULT):
            raise TrialLogError("the holdout already has a result")
        self._append(
            {
                "kind": RecordKind.HOLDOUT_RESULT.value,
                "trial_id": HOLDOUT_TRIAL,
                "at": self.clock(),
                **dict(payload),
            }
        )

    def record_report(self, payload: Mapping[str, Any]) -> None:
        """Every report built is logged, so a verdict cannot be re-rolled out of sight."""
        self._append(
            {
                "kind": RecordKind.REPORT.value,
                "trial_id": REPORT_ENTRY,
                "at": self.clock(),
                **dict(payload),
            }
        )

    def last_report(self) -> dict[str, Any] | None:
        reports = [r for r in self.records() if r["kind"] == RecordKind.REPORT.value]
        return reports[-1] if reports else None
