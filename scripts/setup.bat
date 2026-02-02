@echo off
REM Setup script for Relaxation Agent (Windows)

echo Setting up Relaxation Agent...

REM Change to project root directory (parent of scripts folder)
cd /d "%~dp0.."
echo Current directory: %CD%

REM Create virtual environment
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

REM Download NLTK data
echo Downloading NLTK data...
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

REM Create necessary directories
echo Creating directories...
if not exist "data\raw" mkdir data\raw
if not exist "data\processed" mkdir data\processed
if not exist "models" mkdir models
if not exist "logs" mkdir logs

REM Copy environment file
if not exist ".env" (
    if exist ".env.example" (
        echo Creating .env file...
        copy .env.example .env
        echo Please edit .env with your configuration
    ) else (
        echo Warning: .env.example not found. Creating basic .env file...
        echo # Redis Configuration > .env
        echo REDIS_HOST=localhost >> .env
        echo REDIS_PORT=6379 >> .env
        echo API_HOST=0.0.0.0 >> .env
        echo API_PORT=8000 >> .env
    )
)

echo Setup complete!
echo Next steps:
echo 1. Edit .env with your configuration
echo 2. Start Redis: redis-server
echo 3. Run tests: pytest tests\ -v
echo 4. Start API: python -m src.api.main

pause

