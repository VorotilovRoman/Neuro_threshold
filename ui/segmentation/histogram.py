"""Histogram canvas and rendering shared across segmentation methods."""
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class HistogramCanvas(FigureCanvasQTAgg):
    def __init__(self):
        super().__init__(Figure(figsize=(5, 2)))
        self.axes = self.figure.add_subplot(111)


def update_histogram(owner, gray, threshold=None):
    axes = owner.hist_ax
    axes.clear()
    axes.hist(gray.ravel(), bins=256, range=(0, 256), color='black', alpha=0.7)
    axes.set_title('Grayscale Histogram')
    axes.set_xlabel('Pixel intensity')
    axes.set_ylabel('Frequency')
    if threshold is not None:
        axes.axvline(x=threshold, color='red', linestyle='--', linewidth=1.5, label=f'Threshold: {threshold}')
        axes.legend()
    owner.hist_canvas.draw_idle()


def update_current(owner):
    gray = getattr(owner, 'current_gray', None)
    if gray is None and getattr(owner, 'gray_images', None):
        gray = owner.gray_images[owner.current_index]
    if gray is not None:
        update_histogram(owner, gray, getattr(owner, 'current_threshold', None))
