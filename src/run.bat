@echo off
:: Define the Python script to run
set PYTHON_SCRIPT=C:\Users\Rohit\PycharmProjects\pyVertexModel\src\pyVertexModel\main.py
:: Get the project directory (two levels up from this script)
set "SCRIPT_DIR=%~dp0"
set "PROJECT_DIR=%SCRIPT_DIR%\..\.."
cd /d "%PROJECT_DIR%"
set "PROJECT_DIR=%CD%"
:: Add the project directory to PYTHONPATH
set PYTHONPATH=%PROJECT_DIR%;%PYTHONPATH%
:: Activate Conda environment (replace ‘myenv’ with your env name)
call conda activate pyVertexModel
:: Run the Python script
python "%PYTHON_SCRIPT%"
:: Optional: Pause to see output (remove if not needed)
pause