@echo off
cd /d "%~dp0.."
call scripts\with_3dgs.cmd python -u external\RaDe-GS\train.py -s data\nerf_synthetic\lego -m outputs\lego_radegs --eval -w -r 1 --data_device cpu %*
exit /b %errorlevel%
