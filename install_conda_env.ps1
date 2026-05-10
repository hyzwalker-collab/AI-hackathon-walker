$ErrorActionPreference = "Stop"

$envName = "ai-textbook-agent"
$condaRoot = "D:\app\Anaconda3\anaconda3install"
$logDir = Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "logs"
$logPath = Join-Path $logDir "install_conda_env.log"
$env:Path = "$condaRoot\condabin;$condaRoot;$condaRoot\Scripts;$condaRoot\Library\bin;$env:Path"

if (!(Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir | Out-Null
}

Start-Transcript -Path $logPath -Force | Out-Null

$packages = @(
    "streamlit",
    "pymupdf",
    "python-dotenv",
    "pandas",
    "numpy",
    "networkx",
    "pyvis",
    "scikit-learn",
    "sentence-transformers",
    "faiss-cpu",
    "openai",
    "jieba"
)

Write-Host "Creating conda environment: $envName" -ForegroundColor Cyan
conda create -n $envName -y -c conda-forge --solver libmamba python=3.11
if ($LASTEXITCODE -ne 0) { throw "Failed to create environment $envName" }

Write-Host ""
Write-Host "Installing packages with conda-forge..." -ForegroundColor Cyan
conda install -n $envName -y -c conda-forge --solver libmamba @packages
if ($LASTEXITCODE -ne 0) { throw "Failed to install packages into $envName" }

Write-Host ""
Write-Host "Verifying imports..." -ForegroundColor Cyan
$verify = @'
import streamlit
import fitz
import dotenv
import pandas
import numpy
import networkx
import pyvis
import sklearn
import sentence_transformers
import faiss
import openai
import jieba
print("IMPORT_OK")
'@

conda run -n $envName python -c $verify
if ($LASTEXITCODE -ne 0) { throw "Import verification failed for $envName" }

Write-Host ""
Write-Host "Environment ready: $envName" -ForegroundColor Green

Stop-Transcript | Out-Null
