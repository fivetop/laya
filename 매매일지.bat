@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================================
echo    Trade log viewer (smart-long: gate 0.42 / CW / SELLSIG)
echo ============================================================
echo.
echo   usage:
echo     (just run)                          ask a date (Enter = all)
echo     trade_log.bat 2025-01-10            only that day
echo     trade_log.bat --start 2025-01-06 --end 2025-01-10
echo     trade_log.bat --code 089790         only that stock
echo     trade_log.bat --reason SELLSIG      only sell-signal exits
echo     trade_log.bat --limit 0             show all rows
echo.

python -X utf8 live\trade_log.py %*
echo.
pause
