# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(
    ['face_matcher_app.py'],
    pathex=[],
    binaries=[],
    datas=[('c:\\users\\saul_t_lode\\appdata\\local\\packages\\pythonsoftwarefoundation.python.3.9_qbz5n2kfra8p0\\localcache\\local-packages\\python39\\site-packages\\face_recognition_models\\models', 'face_recognition_models\\models'), ('C:\\Users\\Saul_T_Lode\\AppData\\Local\\Packages\\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\\LocalCache\\local-packages\\Python39\\site-packages\\PyQt5\\Qt5\\plugins', 'PyQt5\\Qt\\plugins'), ('C:\\Users\\Saul_T_Lode\\AppData\\Local\\Packages\\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\\LocalCache\\local-packages\\Python39\\site-packages\\PyQt5\\Qt5\\bin', 'PyQt5\\Qt\\bin'), ('styles', 'styles')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='face_matcher_app',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
