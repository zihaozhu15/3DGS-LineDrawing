@echo off
cd /d "%~dp0.."
call scripts\with_3dgs.cmd python -u viewer\app.py %*
exit /b %errorlevel%
