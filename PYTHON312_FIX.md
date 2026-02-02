# Python 3.12 Compatibility Fix

## Issue
Python 3.12 removed `pkgutil.ImpImporter`, which causes build errors with older package versions (especially numpy 1.24.3).

## Solution

The `requirements.txt` has been updated with Python 3.12 compatible versions.

### Quick Fix

1. **Upgrade pip and setuptools first:**
   ```cmd
   python -m pip install --upgrade pip setuptools wheel
   ```

2. **Install dependencies:**
   ```cmd
   pip install -r requirements.txt
   ```

### If you still have issues:

**Option 1: Install numpy separately first**
```cmd
pip install numpy>=1.26.0
pip install -r requirements.txt
```

**Option 2: Use Python 3.11 (if available)**
If you have Python 3.11 installed, you can use that instead:
```cmd
py -3.11 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**Option 3: Install packages without strict version pinning**
```cmd
pip install fastapi uvicorn[standard] pydantic python-dotenv
pip install redis hiredis
pip install sqlalchemy aiosqlite
pip install librosa soundfile numpy>=1.26.0
pip install torch torchaudio scikit-learn pandas
pip install nltk transformers
pip install httpx aiohttp
pip install pytest pytest-asyncio pytest-cov
pip install python-multipart pyyaml
```

## Verified Compatible Versions for Python 3.12

- numpy: >= 1.26.0
- torch: >= 2.1.0 (should work, but may need specific version)
- All other packages should work with the versions specified

## Check Your Python Version

```cmd
python --version
```

If you see Python 3.12.x, you need the updated requirements.txt.

## Alternative: Use Pre-built Wheels

If building from source fails, pip should automatically try to use pre-built wheels. Make sure you have the latest pip:

```cmd
python -m pip install --upgrade pip
```

