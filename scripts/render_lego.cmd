@echo off
cd /d "%~dp0.."
call scripts\with_3dgs.cmd python external\RaDe-GS\render.py -m outputs\lego_radegs --skip_train %*
exit /b %errorlevel%
