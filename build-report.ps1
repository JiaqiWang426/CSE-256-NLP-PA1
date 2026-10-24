param(
    [string]$Source = "Report.tex"
)

$ErrorActionPreference = "Stop"
$texSource = Get-Item -LiteralPath (Join-Path $PSScriptRoot $Source)
$buildDirectory = Join-Path $texSource.DirectoryName "build"
New-Item -ItemType Directory -Path $buildDirectory -Force | Out-Null

$engine = Join-Path $env:LOCALAPPDATA "Programs/MiKTeX/miktex/bin/x64/xelatex.exe"
if (-not (Test-Path -LiteralPath $engine)) {
    $engine = (Get-Command xelatex -ErrorAction Stop).Source
}

Push-Location -LiteralPath $texSource.DirectoryName
try {
    foreach ($pass in 1..2) {
        Write-Host "XeLaTeX pass $pass/2"
        & $engine "-synctex=1" "-interaction=nonstopmode" "-file-line-error" `
            "-halt-on-error" "-output-directory=$buildDirectory" $texSource.FullName
        if ($LASTEXITCODE -ne 0) {
            throw "XeLaTeX failed. Check $buildDirectory/$($texSource.BaseName).log"
        }
    }
} finally {
    Pop-Location
}

$pdfPath = Join-Path $buildDirectory "$($texSource.BaseName).pdf"
if (-not (Test-Path -LiteralPath $pdfPath)) {
    throw "No PDF was generated. The document may have an empty body."
}
Write-Host "PDF ready: $pdfPath"
