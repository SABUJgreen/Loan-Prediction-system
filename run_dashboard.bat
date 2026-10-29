@echo off
echo ========================================================
echo  Starting LoanGuard AI - Loan Prediction Dashboard
echo ========================================================
echo.

where py >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo Launching with Python 3.12...
    py -3.12 -m streamlit run app.py
    goto end
)

where python >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo Launching with default Python...
    python -m streamlit run app.py
    goto end
)

echo Python was not found in PATH. Please install Python 3.10+ and Streamlit.
pause

:end

