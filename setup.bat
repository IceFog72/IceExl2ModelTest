@echo off

:: Creates a venv if it doesn't exist and runs the start script for requirements upgrades
:: This is intended for users who want to start the API and have everything upgraded and installed


call conda deactivate
call conda activate IceExl2ModelTest
:: Don't create a venv if a conda environment is active
if exist "%CONDA_PREFIX%" (
    echo It looks like you're in a conda environment. Skipping venv check.
) else (
    if not exist "venv\" (
        echo Venv doesn't exist! Creating one for you.
        python -m venv venv
    )

    call .\venv\Scripts\activate.bat
)
call cd /d %~dp0
:: Call the python script with batch args
call python start.py %*
call pause