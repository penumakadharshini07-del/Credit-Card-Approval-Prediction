$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Test-Python {
    param([string]$PythonPath, [string[]]$ExtraArgs = @())
    try {
        & $PythonPath @ExtraArgs -c "import sys; print(sys.version)" *> $null
        return $LASTEXITCODE -eq 0
    }
    catch {
        return $false
    }
}

function Get-SystemPythonCommand {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        try {
            py -3 -c "import sys; print(sys.version)" *> $null
            if ($LASTEXITCODE -eq 0) {
                return @{ Command = "py"; Args = @("-3") }
            }
        }
        catch {}
    }

    if (Get-Command python -ErrorAction SilentlyContinue) {
        try {
            python -c "import sys; print(sys.version)" *> $null
            if ($LASTEXITCODE -eq 0) {
                return @{ Command = "python"; Args = @() }
            }
        }
        catch {}
    }

    return $null
}

function Write-PythonInstallHelp {
    Write-Host ""
    Write-Host "Python is required before this project can run." -ForegroundColor Red
    Write-Host "Install Python 3.11 or newer, then run the review command again."
    Write-Host ""
    Write-Host "Recommended terminal command:" -ForegroundColor Yellow
    Write-Host "winget install -e --id Python.Python.3.11" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "If winget is not available, install it from https://www.python.org/downloads/"
    Write-Host "During setup, tick 'Add python.exe to PATH'."
    Write-Host ""
    Write-Host "After installing Python, close and reopen the terminal, then run:"
    Write-Host "powershell -ExecutionPolicy Bypass -File .\Review-Project.ps1" -ForegroundColor Yellow
}

Set-Location -Path $PSScriptRoot
$projectTemp = "C:\tmp\credit-card-approval-temp"
New-Item -ItemType Directory -Force -Path $projectTemp | Out-Null
$env:TEMP = $projectTemp
$env:TMP = $projectTemp

Write-Host "Credit Card Approval Prediction - Project Review" -ForegroundColor Green
Write-Host "This will prepare the environment, install packages, run tests, and start the website."

$venvDir = Join-Path $PSScriptRoot ".web_venv"
$venvPython = Join-Path $venvDir "Scripts\python.exe"
$pythonCommand = $null

if ((Test-Path $venvPython) -and (Test-Python $venvPython)) {
    $pythonCommand = $venvPython
}
else {
    $systemPython = Get-SystemPythonCommand
    if (-not $systemPython) {
        if (Test-Path $venvPython) {
            Write-Host ""
            Write-Host "A local .web_venv was found, but its Python is not runnable." -ForegroundColor Yellow
            Write-Host "This often happens when the project is inside OneDrive and the virtual environment became a cloud placeholder."
            Write-Host "The review script will rebuild .web_venv automatically after Python is installed."
        }
        Write-PythonInstallHelp
        exit 1
    }

    if (Test-Path $venvDir) {
        $backupName = ".web_venv_broken_{0}" -f (Get-Date -Format "yyyyMMdd_HHmmss")
        Write-Step "Existing virtual environment is not usable. Renaming it to $backupName"
        Rename-Item -Path $venvDir -NewName $backupName
    }

    Write-Step "Creating a fresh virtual environment"
    & $systemPython["Command"] @($systemPython["Args"] + @("-m", "venv", ".web_venv"))
    if ($LASTEXITCODE -ne 0) {
        throw "Virtual environment creation failed."
    }
    $pythonCommand = $venvPython
}

Write-Step "Installing required packages"
& $pythonCommand -m pip install --upgrade pip
& $pythonCommand -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    throw "Package installation failed. Check your internet connection, then run this script again."
}

Write-Step "Checking required project files"
$requiredFiles = @(
    "app.py",
    "run.py",
    "predict.py",
    "preprocessing.py",
    "requirements.txt",
    "data\application_record.csv",
    "data\credit_record.csv",
    "models\best_model.pkl",
    "models\model_metadata.pkl"
)

$missing = @()
foreach ($file in $requiredFiles) {
    if (-not (Test-Path (Join-Path $PSScriptRoot $file))) {
        $missing += $file
    }
}

if ($missing.Count -gt 0) {
    Write-Host "Missing required files:" -ForegroundColor Yellow
    $missing | ForEach-Object { Write-Host " - $_" }
    if ($missing -contains "models\best_model.pkl") {
        Write-Host "Run '.\.web_venv\Scripts\python.exe train.py' after dependencies install to rebuild the model."
    }
}
else {
    Write-Host "All required files are present." -ForegroundColor Green
}

Write-Step "Running automated tests"
& $pythonCommand -m pytest -q
if ($LASTEXITCODE -ne 0) {
    throw "Tests failed. Fix the reported issue, then run this script again."
}

Write-Step "Starting the website"
Write-Host "Open this address in your browser: http://127.0.0.1:5000" -ForegroundColor Green
Write-Host "Use Ctrl+C in this terminal to stop the website."
& $pythonCommand run.py
