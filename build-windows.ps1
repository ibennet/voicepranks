# Build the Windows standalone bundle (dist\minion-voice\) and zip it.
#
# Must run on Windows (PyInstaller can't cross-compile). From PowerShell:
#   .\build-windows.ps1
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$Python = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$Venv = ".venv-build"
# The bundle folder name is owned solely by minion-voice.spec; this script
# discovers whatever PyInstaller produced rather than re-hardcoding the name.

Write-Host "==> Creating clean build venv ($Venv)"
if (Test-Path $Venv) { Remove-Item -Recurse -Force $Venv }
& $Python -m venv $Venv
$VenvPy = Join-Path $Venv "Scripts\python.exe"

& $VenvPy -m pip install --quiet --upgrade pip
& $VenvPy -m pip install --quiet -r requirements.txt -r requirements-build.txt

Write-Host "==> Running PyInstaller"
if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
if (Test-Path "dist") { Remove-Item -Recurse -Force "dist" }  # clean slate for the discovery below
& $VenvPy -m PyInstaller minion-voice.spec --noconfirm

# Locate the one-folder bundle PyInstaller produced (name comes from the spec).
$Built = @(Get-ChildItem -Path "dist" -Directory)
if ($Built.Count -ne 1) { throw "Expected exactly one bundle folder in dist/, found $($Built.Count)" }

Write-Host "==> Staging zip (folder + INSTALL.txt)"
$Stage = "dist\minion-voice-windows"
New-Item -ItemType Directory -Path $Stage | Out-Null
Move-Item $Built[0].FullName $Stage  # move (not copy) — the zip is the deliverable
Copy-Item "INSTALL.txt" $Stage
Compress-Archive -Path "$Stage\*" -DestinationPath "$Stage.zip"

Write-Host "==> Done: $Stage.zip"
