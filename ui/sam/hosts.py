"""Small adapters: SAM workflow is shared; each tab owns its result destination."""

from runtime_torch import prepare_windows_torch

prepare_windows_torch()

from utils_new.annotations.adapters import from_legacy
from utils_new.annotations.model import ImageSize
from utils_new.segmentation.backends.sam import SamResult
from utils_new.segmentation.result import SegmentationResult
from utils_new.images.morphology import apply_morphology


class SamHost:
    confirm_replace = False

    def __init__(self, owner, view):
        self.owner, self.panel, self.view = owner, owner.sam_panel, view

    def identity(self):
        owner = self.owner
        path = str(owner.image_paths[owner.current_index]) if 0 <= owner.current_index < len(owner.image_paths) else None
        return id(owner._lazy_cache), getattr(owner, '_data_revision', 0), owner.current_index, path

    def can_start(self):
        return True

    def can_accept(self, result, annotations):
        return bool(annotations)

    def frame_changed(self):
        pass

    def model_changed(self):
        pass


class LabelerSamHost(SamHost):
    confirm_replace = True

    def __init__(self, owner):
        super().__init__(owner, owner.image_view)

    def bind(self, controller):
        def changed():
            controller.provider_changed()
            self.owner.auto_stack.setCurrentIndex(self.owner.auto_backend_combo.currentIndex())
        self.owner.auto_backend_combo.currentIndexChanged.connect(changed)

    def image_ready(self):
        return self.owner._current_frame_ready()

    def image(self):
        return self.owner._lazy_cache.get(self.owner.current_index)['image']

    def reset_tool(self):
        self.owner.reset_drawing_tool()

    def selected_annotation(self):
        index = self.view.selected_index
        if 0 <= index < len(self.owner.current_annotations):
            return from_legacy(self.owner.current_annotations[index], ImageSize(self.owner.img_w, self.owner.img_h))
        return None

    def busy_changed(self, busy):
        self.owner.auto_group.setEnabled(not busy)

    def apply_result(self, result, annotations, replace):
        self.owner._apply_auto_annotations(annotations, replace)

    def accepted_message(self, count):
        return f'Принято объектов: {count}. Сохраните разметку кнопкой Save Labels.'


class DeepLearningSamHost(SamHost):
    def __init__(self, owner):
        super().__init__(owner, owner.original_view)
        self.result = self.processed = None
        self.class_id, self.min_area = 0, 10

    @property
    def active(self):
        return self.owner.model_combo.currentText() == 'SAM'

    def bind(self, controller):
        self.panel.replace.hide()
        self.panel.accept.setText('Применить маску')
        self.panel.kind.hide()
        self.panel.kind_label.hide()
        self.owner.draw_combo.currentIndexChanged.connect(self.sync_kind)

    def sync_kind(self, *_):
        index = self.panel.kind.findData(self.owner.draw_combo.currentData())
        if index >= 0:
            self.panel.kind.setCurrentIndex(index)

    def image_ready(self):
        frame = getattr(self.owner, '_current_frame', None)
        return (frame is not None and self.owner.original_image is not None
                and frame.key[:2] == (self.owner._data_revision, self.owner.current_index)
                and self.identity()[-1] == frame.path)

    def image(self):
        return self.owner.original_image

    def reset_tool(self):
        self.view.set_drawing_tool(None)

    def selected_annotation(self):
        index = getattr(self.owner, '_focused_object_index', -1)
        if self.image_ready() and 0 <= index < len(self.owner.current_objects_full):
            h, w = self.image().shape[:2]
            return from_legacy(self.owner.current_objects_full[index], ImageSize(w, h))
        return None

    def can_start(self):
        return not self.owner.jobs.busy

    def can_accept(self, result, annotations):
        return True  # Empty binary masks are a valid segmentation result too.

    def busy_changed(self, busy):
        for widget in (self.owner.model_combo, self.owner.preset_combo,
                       self.owner.save_preset_btn, self.owner.delete_preset_btn):
            widget.setEnabled(not busy)

    def reset_result(self):
        self.result = self.processed = None
        if self.active:
            self.owner.prediction_mask = None
            self.owner._raw_mask = self.owner._raw_signature = None
            self.owner.result_state = SegmentationResult()

    def frame_changed(self):
        self.reset_result()

    def model_changed(self):
        self.reset_result()
        if self.active:
            self.owner.schedule_update()

    def apply_result(self, result, annotations, replace):
        self.result = result
        self.class_id, self.min_area = self.panel.class_id.value(), self.panel.area.value()
        self.owner.prediction_mask = result.combined_mask()
        self.owner._raw_signature = None
        self.owner.display_current_image()

    def postprocess(self, raw, frame):
        widget = self.owner.morphology_widget
        if self.active and self.result is not None:
            self.processed = self.result.postprocessed(widget.get_steps(), widget.kernel_shape())
            return self.processed.combined_mask()
        return apply_morphology(raw, widget.get_steps(), shape=widget.kernel_shape())

    def objects(self):
        if not self.active or self.processed is None:
            return None
        result = self.processed
        if self.owner.invert_checkbox.isChecked():
            # Invert the combined mask once; complement is a semantic background.
            result = SamResult.from_binary_mask(self.owner.final_mask)
        return result.annotations(class_id=self.class_id, kind=self.owner.draw_combo.currentData(),
                                  min_area=self.min_area, use_hull=self.owner.hull_checkbox.isChecked())

    def accepted_message(self, count):
        return 'Маска применена. Проверьте морфологию, отметьте объекты в списке и сохраните разметку.'
