@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ============================================================
echo    Laya live pipeline - self tests
echo ============================================================
echo.

set PASS=0
set FAIL=0
set SKIP=0

echo [A] Fast tests (MongoDB not required)
echo ------------------------------------------------------------
echo.

echo [1/6] Trader golden check  (live\selftest_replay.py)
python -X utf8 live\selftest_replay.py
if !errorlevel!==0 (echo    RESULT: PASS & set /a PASS+=1) else (echo    RESULT: FAIL & set /a FAIL+=1)
echo.

echo [2/6] State check  (live\selftest_state.py --source parquet)
python -X utf8 live\selftest_state.py --source parquet
if !errorlevel!==0 (echo    RESULT: PASS & set /a PASS+=1) else (echo    RESULT: FAIL & set /a FAIL+=1)
echo.

echo [3/6] Replay 100 days  (replay --source bars, golden +10.44 pct)
python -X utf8 live\replay.py --source bars --start-date 2025-01-06 --end-date 2025-07-23
if !errorlevel!==0 (echo    RESULT: PASS & set /a PASS+=1) else (echo    RESULT: FAIL & set /a FAIL+=1)
echo.

echo [4/6] Replay eval 20 days  (replay --source bars, +8.05 pct)
python -X utf8 live\replay.py --source bars --start-date 2025-06-25 --end-date 2025-07-23
if !errorlevel!==0 (echo    RESULT: PASS & set /a PASS+=1) else (echo    RESULT: FAIL & set /a FAIL+=1)
echo.

echo [B] Tests needing MongoDB
echo ------------------------------------------------------------
echo.

tasklist /fi "imagename eq mongod.exe" 2>nul | find /i "mongod.exe" >nul
if !errorlevel! neq 0 goto nomongo

echo mongod is running - proceed
echo.
echo [5/6] Bar aggregation check  (live\selftest_bars.py)
python -X utf8 live\selftest_bars.py
if !errorlevel!==0 (echo    RESULT: PASS & set /a PASS+=1) else (echo    RESULT: FAIL & set /a FAIL+=1)
echo.

echo [6/6] End-to-end tick replay  (replay --source ticks, about 60s)
python -X utf8 live\replay.py --source ticks --start-date 2025-06-24 --trade-start 2025-06-25 --end-date 2025-07-23
if !errorlevel!==0 (echo    RESULT: PASS & set /a PASS+=1) else (echo    RESULT: FAIL & set /a FAIL+=1)
echo.
goto summary

:nomongo
echo mongod is not running - skipping [5],[6]  (start MongoDB and rerun)
set /a SKIP+=2
echo.

:summary
echo ============================================================
echo    Summary:  PASS !PASS!   FAIL !FAIL!   SKIP !SKIP!
echo ============================================================
echo.
pause
