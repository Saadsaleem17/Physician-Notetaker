# Setup Script for Medical Transcription NLP Pipeline

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Medical Transcription NLP Pipeline Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python version
Write-Host "Checking Python version..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
Write-Host "Found: $pythonVersion" -ForegroundColor Green

# Create virtual environment
Write-Host "`nCreating virtual environment..." -ForegroundColor Yellow
python -m venv venv
Write-Host "Virtual environment created!" -ForegroundColor Green

# Activate virtual environment
Write-Host "`nActivating virtual environment..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host "`nUpgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install requirements
Write-Host "`nInstalling required packages..." -ForegroundColor Yellow
pip install -r requirements.txt

# Download spaCy model
Write-Host "`nDownloading spaCy language model..." -ForegroundColor Yellow
python -m spacy download en_core_web_sm

# Create output directory
Write-Host "`nCreating output directory..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "output" | Out-Null
Write-Host "Output directory created!" -ForegroundColor Green

# Test installation
Write-Host "`nTesting installation..." -ForegroundColor Yellow
python -c "import spacy, transformers, torch; print('All packages imported successfully!')"

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "`nNext steps:" -ForegroundColor Yellow
Write-Host "1. Run: python main.py" -ForegroundColor White
Write-Host "2. Or open: Medical_NLP_Pipeline.ipynb" -ForegroundColor White
Write-Host ""
