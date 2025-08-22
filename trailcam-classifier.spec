# -*- mode: python ; coding: utf-8 -*-

import sys
from PyInstaller.utils.hooks import collect_data_files

block_cipher = None


a = Analysis(
    ['src/trailcam_classifier/gui.py'],
    pathex=['src'],
    binaries=[],
    datas=[],
    hiddenimports=['torchvision', 'sklearn', 'PIL'],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Add data files for PySide6
a.datas += collect_data_files('PySide6')

# Add the model and class names file.
# The user will need to replace these with the actual paths to their files.
# The second element of the tuple is the destination folder in the bundle.
a.datas += [('image_classifier_model.pth', '.'),
            ('class_names.txt', '.')]


pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='trailcam-classifier',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='trailcam-classifier',
)
app = BUNDLE(
    coll,
    name='trailcam-classifier.app',
    icon=None,
    bundle_identifier=None,
)
