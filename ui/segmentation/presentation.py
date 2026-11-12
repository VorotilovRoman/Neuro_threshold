"""Frame/stage presentation shared by all five segmentation methods."""
from pathlib import Path

import cv2

from ui.image_conversion import numpy_to_qpixmap
from ui.segmentation import histogram, results, session
from ui.segmentation.panels import STAGES
from utils_new.annotations.modes import supports_hull
from utils_new.morphology import apply_morphology
from utils_new.segmentation.postprocessing import extract_objects
from utils_new.segmentation.result import (
    ResultStatus,
    SegmentationResult,
    binary_mask,
    masked_preview,
)

STATUS_TEXT={ResultStatus.NO_IMAGE:'Нет изображения',ResultStatus.NOT_RUN:'Маска не вычислена',
             ResultStatus.RUNNING:'Вычисление…',ResultStatus.READY:'Результат готов',ResultStatus.ERROR:'Ошибка обработки'}


def show_image(view,image,annotations=()):
    view.set_selected_index(-1)
    if image is None:
        view.set_annotations([],0,0);view.set_pixmap(numpy_to_qpixmap(None));view._segmentation_base=None
    else:
        h,w=image.shape[:2];view.set_annotations(list(annotations),w,h);view.set_pixmap(numpy_to_qpixmap(image))


def set_status(owner,status,error=None):
    owner.result_status_label.setText(STATUS_TEXT[status]+(f': {error}' if error else ''))
    owner.save_button.setEnabled(status==ResultStatus.READY)


def show_unavailable(owner,status,error=None):
    owner.result_state=SegmentationResult(status,error=error);owner.final_mask=None
    owner.current_objects_full=[];owner.current_selected_indices=[];owner._focused_object_index=-1
    owner.current_base_image=getattr(owner,'original_image',None)
    for name,_ in STAGES[owner.segmentation_kind]:
        if name not in ('original_view','annotated_view'):
            view=getattr(owner,name);show_image(view,None)
            if owner.current_base_image is not None:
                h,w=owner.current_base_image.shape[:2];view.set_image_data(w,h)
    show_image(owner.annotated_view,owner.current_base_image)
    owner.object_list.clear();owner.coord_text.clear()
    set_status(owner,status,error)


def no_frame(owner,status=ResultStatus.NO_IMAGE,error=None):
    owner.original_image=owner.current_gray=None;owner._raw_mask=None;owner._raw_signature=None;owner._frame_key=None
    for name in owner.segmentation_view_names:show_image(getattr(owner,name),None)
    owner.hist_ax.clear();owner.hist_canvas.draw_idle()
    owner.info_label.setText('Нет изображения')
    show_unavailable(owner,status,error)


def geometry_changed(owner,*_):
    mode=owner.draw_combo.currentData();owner.hull_checkbox.setVisible(supports_hull(mode))
    if owner.result_state.status!=ResultStatus.READY or owner.final_mask is None:
        return
    owner.current_objects_full=extract_objects(owner.current_base_image,owner.final_mask,mode,owner.hull_checkbox.isChecked())[1]
    owner.update_object_list()


def postprocess(owner,frame):
    raw=owner._raw_mask
    if raw is None:
        state=owner.result_state
        show_unavailable(owner,state.status if state.status in (ResultStatus.ERROR,ResultStatus.RUNNING) else ResultStatus.NOT_RUN,state.error)
        return
    processed=apply_morphology(raw,owner.morphology_widget.get_steps(),shape=owner.morphology_widget.kernel_shape())
    final=processed
    finalize=getattr(owner,'_finalize_mask',None)
    if finalize is not None:
        final=finalize(processed.copy(),frame)
    if owner.invert_checkbox.isChecked():final=cv2.bitwise_not(final)
    show_image(owner.morph_view,processed if finalize is not None else final)
    if hasattr(owner,'filled_view'):show_image(owner.filled_view,final)
    owner.final_mask=final;owner.result_state=SegmentationResult(ResultStatus.READY,final)
    owner._reported_error=None
    owner.current_base_image=masked_preview(frame.image,final)
    owner.annotated_view._segmentation_base=None
    geometry_changed(owner)
    set_status(owner,ResultStatus.READY)


def display(owner):
    from ui.segmentation import errors
    frame=None
    try:
        frame=session.current_frame(owner)
        if frame is None:
            no_frame(owner);return
        changed=frame.key!=getattr(owner,'_frame_key',None)
        if changed:
            owner._raw_mask=None;owner._raw_signature=None
            owner.result_state=SegmentationResult()
        owner._frame_key=frame.key;owner._current_frame=frame
        owner.original_image=frame.image;owner.current_gray=frame.gray
        results.prepare_frame(owner,frame.gray)
        owner.info_label.setText(f'Изображение {owner.current_index+1} из {len(owner.image_paths)} — {Path(frame.path).name}')
        owner.nav_widget.set_current_index(owner.current_index,len(owner.image_paths))
        for name,_ in STAGES[owner.segmentation_kind]:
            getattr(owner,name).set_suggested_save_name(f'{owner.segmentation_kind}_{name}_{Path(frame.path).stem}')
        show_image(owner.original_view,frame.image)
        hook=getattr(owner,'_present_input_frame',None)
        if hook is not None:hook(frame)
        signature=owner._mask_signature(frame)
        if signature!=owner._raw_signature:
            mask,stages=owner._compute_mask(frame)
            owner._raw_mask=None if mask is None else binary_mask(mask,frame.gray.shape)
            owner._raw_signature=signature
            for name,image in stages.items():show_image(getattr(owner,name),image)
        histogram.update_histogram(owner,frame.gray,getattr(owner,'current_threshold',None))
        postprocess(owner,frame)
        session.prefetch(owner)
    except Exception as exc:
        if frame is None:
            no_frame(owner,ResultStatus.ERROR,str(exc))
        errors.report(owner,'Сегментация',exc)
