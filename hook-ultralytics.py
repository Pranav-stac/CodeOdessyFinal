"""
PyInstaller hook for ultralytics
"""
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Collect all ultralytics submodules
hiddenimports = collect_submodules('ultralytics')

# Collect data files
datas = collect_data_files('ultralytics')

