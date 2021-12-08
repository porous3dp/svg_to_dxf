@echo off
echo "Building svg_to_dxf..."

cd src
pyinstaller svg_to_dxf.py

rmdir build
move dist ..\dist

cd ..
pwd
