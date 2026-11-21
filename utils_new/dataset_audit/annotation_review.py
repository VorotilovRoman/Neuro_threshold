"""Shared annotation-overlap review for editors, preparation and analysis. No Qt."""

from utils_new.annotation_processing import find_strong_overlaps

from .objects import image_size, load_pair_annotations, read_objects


class AnnotationReviewJob:
    def __init__(
        self, pairs, dataset_type=0, input_format="auto", options=None, memory=None
    ):
        self.pairs = pairs
        self.dataset_type = dataset_type
        self.input_format = input_format
        self.options = options or {}
        self.memory = memory or {}
        self.cancelled = False
        self.progress = lambda *args: None
        self.log = lambda message: None

    def cancel(self):
        self.cancelled = True

    def run(self):
        records, errors = {}, {}
        for index, pair in enumerate(self.pairs):
            if self.cancelled:
                return {"cancelled": True, "records": {}, "errors": {}}
            path = str(pair[0])
            try:
                if path in self.memory:
                    annotations, size = self.memory[path]
                elif self.dataset_type == 0:
                    size = image_size(path)
                    annotations = (
                        load_pair_annotations(path, pair[1], self.input_format)
                        if pair[1]
                        else []
                    )
                elif self.dataset_type == 1:
                    size, objects = read_objects(pair, 1, self.input_format)
                    annotations = [
                        o.annotation for o in objects if o.annotation is not None
                    ]
                else:
                    size, annotations = image_size(path), []
                records[path] = find_strong_overlaps(annotations, *size, **self.options)
            except (OSError, ValueError, TypeError, IndexError) as exc:
                errors[path] = str(exc)
            self.progress(index + 1, len(self.pairs))
        return {"records": records, "errors": errors, "cancelled": False}
