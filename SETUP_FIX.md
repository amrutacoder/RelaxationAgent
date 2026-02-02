# Quick Setup Fix

## Issue
The setup script was run from the `scripts` directory instead of the project root, causing:
- `requirements.txt` not found
- NLTK not installed
- `.env.example` not found

## Solution

### Option 1: Run Setup from Project Root (Recommended)

1. **Navigate to project root:**
   ```cmd
   cd "C:\Users\DELL\Desktop\Relaxation Agent"
   ```

2. **Run the setup script:**
   ```cmd
   scripts\setup.bat
   ```

   The script now automatically changes to the project root, so you can run it from anywhere!

### Option 2: Manual Setup (If script still has issues)

1. **Navigate to project root:**
   ```cmd
   cd "C:\Users\DELL\Desktop\Relaxation Agent"
   ```

2. **Create/activate virtual environment:**
   ```cmd
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```cmd
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Download NLTK data:**
   ```cmd
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
   ```

5. **Create directories:**
   ```cmd
   mkdir data\raw
   mkdir data\processed
   mkdir models
   mkdir logs
   ```

6. **Create .env file:**
   Create a file named `.env` in the project root with:
   ```env
   REDIS_HOST=localhost
   REDIS_PORT=6379
   API_HOST=0.0.0.0
   API_PORT=8000
   DATABASE_PATH=./data/relaxation_agent.db
   MODEL_PATH=./models/emotion_classifier.pt
   STRESS_THRESHOLD_HIGH=0.7
   STRESS_THRESHOLD_MEDIUM=0.4
   LOG_LEVEL=INFO
   LOG_FILE=./logs/relaxation_agent.log
   ```

## Verify Setup

After setup, test it:

```cmd
# Activate venv (if not already active)
venv\Scripts\activate

# Test the text prototype
python -m src.milestone_a.text_prototype
```

If this runs without errors, setup is complete!

## Next Steps

1. **Start Redis** (if you have it installed):
   ```cmd
   redis-server
   ```

2. **Start the API:**
   ```cmd
   python -m src.api.main
   ```

3. **Test the API:**
   Open browser to: http://localhost:8000/docs

