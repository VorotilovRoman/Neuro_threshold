"""Shared navigation bindings, keeping each tab's dataset independent."""

def navigate(owner, index, reset=None, only_if_changed=False):
    if not owner.image_paths:
        return
    index = max(0, min(index, len(owner.image_paths)-1))
    if only_if_changed and index == owner.current_index:
        return
    owner.current_index = index
    if reset is not None:
        reset()
    owner.display_current_image()
    owner.nav_widget.set_current_index(owner.current_index, len(owner.image_paths))
    owner.update_navigation_state()


def step(owner, delta, reset=None):
    if owner.image_paths:
        navigate(owner, (owner.current_index+delta) % len(owner.image_paths), reset=reset)


def reset_zooms(owner):
    for name in owner.segmentation_view_names:
        getattr(owner, name).reset_view()


def update_state(owner):
    total = len(owner.image_paths)
    owner.nav_widget.set_prev_enabled(total>0 and owner.current_index>0)
    owner.nav_widget.set_next_enabled(total>0 and owner.current_index<total-1)
