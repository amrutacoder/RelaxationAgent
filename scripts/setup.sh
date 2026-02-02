#!/bin/bash
# Setup script for Relaxation Agent

echo "Setting up Relaxation Agent..."

# Change to project root directory (parent of scripts folder)
cd "$(dirname "$0")/.."
echo "Current directory: $(pwd)"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Download NLTK data
echo "Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Create necessary directories
echo "Creating directories..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p models
mkdir -p logs

# Copy environment file
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        echo "Creating .env file..."
        cp .env.example .env
        echo "Please edit .env with your configuration"
    else
        echo "Warning: .env.example not found. Creating basic .env file..."
        cat > .env << EOF
# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
EOF
    fi
fi

echo "Setup complete!"
echo "Next steps:"
echo "1. Edit .env with your configuration"
echo "2. Start Redis: redis-server"
echo "3. Run tests: pytest tests/ -v"
echo "4. Start API: python -m src.api.main"

