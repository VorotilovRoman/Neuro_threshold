"""Cancellable base checks over a snapshot of selected pairs, without Qt."""

from .checks import DuplicateDetector, LeakageChecker, ObjectDistributionHeatmap
from .quality import QualityJob


class ScanJob:
    def __init__(self, kind, pairs, options=None):
        self.kind = kind
        self.pairs = pairs
        self.options = options or {}
        self.cancelled = False
        self.progress = lambda *args: None
        self.log = lambda message: None
        self.child = None

    def cancel(self):
        self.cancelled = True
        if self.child:
            self.child.cancel()

    def run(self):
        if self.cancelled:
            return {"cancelled": True}
        if self.kind == "quality":
            self.child = QualityJob(self.pairs, self.options.get("quality"))
            self.child.progress = self.progress
            if self.cancelled:
                self.child.cancel()
            result = self.child.run()
            return {"cancelled": True} if self.cancelled else result
        if self.kind == "duplicates":
            result = DuplicateDetector.find_duplicates(
                self.pairs,
                phash_threshold=self.options.get("phash_threshold", 8),
                progress_cb=self.progress,
                cancel_check=lambda: self.cancelled,
            )
            if result.get("cancelled") or self.cancelled:
                return {"cancelled": True}
            result["leak"] = LeakageChecker.check(
                result.get("exact", []) + result.get("near", []),
                self.pairs,
                self.options.get("splits"),
            )
            return result
        if self.kind == "heatmap":
            return ObjectDistributionHeatmap.compute(
                self.pairs,
                dataset_type=self.options.get("dataset_type", 0),
                grid_size=self.options.get("grid_size", 16),
                input_format=self.options.get("input_format", "auto"),
                progress_cb=self.progress,
                cancel_check=lambda: self.cancelled,
            )
        if self.kind == "basic":
            output = {}
            kinds = (
                ("quality", "duplicates", "heatmap")
                if self.options.get("dataset_type", 0) != 2
                else ("quality", "duplicates")
            )
            for step, kind in enumerate(kinds):
                self.child = ScanJob(kind, self.pairs, self.options)
                self.child.progress = lambda done, total, step=step: self.progress(
                    step * len(self.pairs) + done, len(kinds) * len(self.pairs)
                )
                if self.cancelled:
                    return {"cancelled": True}
                value = self.child.run()
                if value.get("cancelled") or self.cancelled:
                    return {"cancelled": True}
                output[kind] = value
            return output
        raise ValueError(f"Неизвестная проверка: {self.kind}")
