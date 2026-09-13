@echo off
set "PATH=C:\Program Files (x86)\Microsoft Visual Studio\Installer;%PATH%"
call D:\Anaconda\condabin\conda.bat activate 3dgs
if errorlevel 1 exit /b %errorlevel%
call "D:\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b %errorlevel%
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "CUDA_PATH=%CUDA_HOME%"
set "PATH=%CUDA_HOME%\bin;%PATH%"
set "TORCH_CUDA_ARCH_LIST=12.0"
set "DISTUTILS_USE_SDK=1"
set "MAX_JOBS=4"
%*
exit /b %errorlevel%
