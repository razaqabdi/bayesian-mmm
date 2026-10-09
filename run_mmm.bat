@echo off
setlocal
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
if not exist "src\fit_mmm.py" cd /d "%USERPROFILE%\Documents\bayesian-mmm"
if not exist "src\fit_mmm.py" (echo Can not find the project. Run this file from Documents\bayesian-mmm. & pause & exit /b 1)
if not exist outputs mkdir outputs
set "LOG=%CD%\outputs\run_log.txt"
set "SETUP_LOG=%CD%\outputs\setup_log.txt"
if exist "%LOG%" del "%LOG%"
if exist "%SETUP_LOG%" del "%SETUP_LOG%"
set "CONDA_ROOT=%USERPROFILE%\miniforge3"

echo ============================================================
echo  Bayesian MMM - one-click setup and run
echo ============================================================

if exist "%CONDA_ROOT%\Scripts\conda.exe" goto have_conda
echo [1/4] Downloading Miniforge (Python + conda)...
curl -L -o "%TEMP%\Miniforge3.exe" https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe
if errorlevel 1 (echo Download failed. Check your internet connection. & pause & exit /b 1)
echo [1/4] Installing Miniforge, this takes a few minutes...
start /wait "" "%TEMP%\Miniforge3.exe" /InstallationType=JustMe /RegisterPython=0 /AddToPath=0 /S /D=%CONDA_ROOT%
if not exist "%CONDA_ROOT%\Scripts\conda.exe" (echo Miniforge install failed. & pause & exit /b 1)

:have_conda
call "%CONDA_ROOT%\Scripts\activate.bat" "%CONDA_ROOT%"

call conda env list | findstr /C:"bayesian-mmm" >nul
if errorlevel 1 goto make_env
call conda run -n bayesian-mmm python -c "import pymc_marketing, streamlit, plotly" >nul 2>&1
if not errorlevel 1 goto have_env
echo Existing environment is incomplete, rebuilding it...
call conda env remove -n bayesian-mmm -y >nul 2>&1

:make_env
echo [2/4] Creating the bayesian-mmm environment (5-10 minutes the first time, window will look idle)...
call conda env create -f environment-model.yml > "%SETUP_LOG%" 2>&1
if errorlevel 1 (
  type "%SETUP_LOG%"
  echo.
  echo Environment setup failed. Tell Claude "it failed" - the log is saved in outputs\setup_log.txt
  pause
  exit /b 1
)

:have_env
call conda activate bayesian-mmm

echo [3/4] Simulating the data, then fitting the Bayesian model three times
echo       (with the lift tests, without them for comparison, and a 13-week holdout test).
echo       This takes 5-15 minutes; the window will look idle.
python -u src\simulate.py > "%LOG%" 2>&1
if errorlevel 1 (
  type "%LOG%"
  echo.
  echo Data simulation failed. Tell Claude "it failed" - the log is saved in outputs\run_log.txt
  pause
  exit /b 1
)
python -u src\fit_mmm.py >> "%LOG%" 2>&1
if errorlevel 1 (
  type "%LOG%"
  echo.
  echo Model fit failed. Tell Claude "it failed" - the log is saved in outputs\run_log.txt
  pause
  exit /b 1
)
type "%LOG%"

echo.
echo [4/4] Opening the app in your browser. Close this window to stop it.
if not exist "%USERPROFILE%\.streamlit" mkdir "%USERPROFILE%\.streamlit"
if not exist "%USERPROFILE%\.streamlit\credentials.toml" (
  > "%USERPROFILE%\.streamlit\credentials.toml" echo [general]
  >> "%USERPROFILE%\.streamlit\credentials.toml" echo email = ""
)
streamlit run app.py
pause
