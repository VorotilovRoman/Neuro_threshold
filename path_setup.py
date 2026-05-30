# path_setup.py
import sys
import os

PROJECT_ROOT = None

def setup_project_path():
    global PROJECT_ROOT
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root = current_dir
    while root != os.path.dirname(root):
        if os.path.exists(os.path.join(root, 'main.py')):
            break
        root = os.path.dirname(root)
    if root not in sys.path:
        sys.path.insert(0, root)
    PROJECT_ROOT = root
    return root

def get_project_root():
    global PROJECT_ROOT
    if PROJECT_ROOT is None:
        return setup_project_path()
    return PROJECT_ROOT