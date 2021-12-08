@echo off
echo "Building svg_to_dxf..."

@REM Go to where the source code lives
cd src

@REM Build the thing
pyinstaller svg_to_dxf.py

@REM Delete the temporary build files
rmdir /s /q build
del /f svg_to_dxf.spec

@REM Todo - Delete all the unneeded binaries included in the bundle...

@REM Copy the binaries to a more easily accessible place
move dist\svg_to_dxf ..\dist
rmdir /s /q dist

@REM Reset the shell to the original location
cd ..
pwd

@REM Copy the license text stuff
copy README.md dist\README.txt
copy legal\ dist\


