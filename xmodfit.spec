# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules

block_cipher = None
homedir='C:/Users/mrinalkb/PycharmProjects/XModFit'

hxraydb=collect_submodules('xraydb')
hutils=collect_submodules('utils')
hchemform=collect_submodules('Chemical_Formula')
hnumba=collect_submodules('numba')
hllvmlite=collect_submodules('llvmlite')
hnnumpy=collect_submodules('numba-numpy')

all_hidden_imports=hxraydb+hutils+hchemform+hnumba+hllvmlite

a = Analysis(['xmodfit.py'],
             pathex=[homedir],
             binaries=[('C:/Users/mrinalkb/Anaconda3/Lib/site-packages/llvmlite/binding/llvmlite.dll','.'),],
             datas=[(homedir+'/Fortran_routines','Fortran_routines'),
             (homedir+'/Function_Details','Function_Details'),
             (homedir+'/Tools','Tools'),
             (homedir+'/UI_Forms','UI_Forms'),
             (homedir+'/Functions','Functions'),
             (homedir+'/functionTemplate.txt','.'),
             (homedir+'/license','.')],
             hiddenimports=all_hidden_imports,
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='xmodfit',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='xmodfit')
