$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
if (-not (Test-Path '.venv/Scripts/python.exe')) {
    python -m venv .venv
    if ($LASTEXITCODE) { throw 'Python environment creation failed' }
    & .venv/Scripts/python.exe -m pip install -r requirements.txt
    if ($LASTEXITCODE) { throw 'Dependency installation failed' }
}
if (-not (Test-Path '.env')) { Copy-Item .env.example .env }
& .venv/Scripts/python.exe -m streamlit run app.py --server.address 127.0.0.1

