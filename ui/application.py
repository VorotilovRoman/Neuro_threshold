"""Shared Qt event-loop lifecycle for main and standalone tabs."""
from PyQt5.QtCore import QCoreApplication, QEvent


def exec_window(window, app):
    """Destroy the widget tree before QApplication/native modules shut down.

    Relying on interpreter finalization leaves Qt/Matplotlib objects in an
    undefined destruction order when multiple tabs retain callbacks/widgets.
    """
    try:
        return app.exec_()
    finally:
        if window.isVisible():
            window.close()
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
