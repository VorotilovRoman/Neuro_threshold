# libs_external.py
# -*- coding: utf-8 -*-

# ========== Стандартные библиотеки ==========
import sys
import os
import json
import re
import time
import random
import hashlib
import shutil
import traceback
import warnings
from pathlib import Path
from datetime import datetime

# ========== Сторонние библиотеки ==========
import cv2
import torch
import torchvision.transforms as T
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import colorsys
import yaml
import joblib
import albumentations as A
import onnxruntime as ort
from ast import literal_eval

import logging

# Ultralytics YOLO
import ultralytics
from ultralytics import YOLO
from ultralytics.data.converter import convert_segment_masks_to_yolo_seg


import networkx as nx
# Scikit-learn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from skimage import segmentation
from skimage.segmentation import slic, find_boundaries
from skimage.util import img_as_float
from sklearn.preprocessing import StandardScaler


from ast import literal_eval


# ========== PyQt5 ==========
from PyQt5.QtCore import (
    Qt, QProcess, QTimer, QSettings, QThread, pyqtSignal, QRectF, QPointF, QObject, QDir, QUrl
)
from PyQt5.QtGui import (
    QCursor, QPixmap, QPainter, QColor, QImage, QWheelEvent, QMouseEvent, QDesktopServices,
    QTransform, QPalette, QFont, QPen, QBrush
)

from PyQt5.QtWidgets import (
    QApplication, QToolButton, QMainWindow, QWidget, QFileSystemModel,
    QRadioButton, QTreeView, QStackedWidget, QTableWidget, QTableWidgetItem,
    QHeaderView, QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QPushButton,
    QLabel, QSlider, QCheckBox, QComboBox, QGroupBox, QTextEdit, QSplitter,
    QSpinBox, QGridLayout, QSizePolicy, QListWidget, QAbstractItemView,
    QListWidgetItem, QFileDialog, QMessageBox, QInputDialog, QProgressDialog,
    QMenu, QAction, QGraphicsView, QSplashScreen, QProgressBar, QDoubleSpinBox,
    QDialogButtonBox, QScrollArea, QFormLayout, QGraphicsScene, QGraphicsPixmapItem,
    QFrame, QLineEdit
)

# ========== Matplotlib ==========
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ========== Печать версий ==========
print(f"OpenCV version: {cv2.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"albumentations version: {A.__version__}")
print(f"Ultralytics YOLO version: {ultralytics.__version__}")
