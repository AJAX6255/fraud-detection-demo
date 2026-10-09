# Deploy the fraud-detection engine to Google Cloud Run from PowerShell
param (
    [Parameter(Position=0)]
    [string]$ProjectId = "",

    [Parameter(Position=1)]
    [string]$Region = "us-central1",

    [Parameter(Position=2)]
    [ValidateSet("api", "streamlit")]
    [string]$ServiceMode = "streamlit"
)

# Auto-load .env if available
if (Test-Path ".env") {
    Write-Host "==> Loading environment variables from .env..." -ForegroundColor DarkGray
    Get-Content ".env" | ForEach-Object {
        if ($_ -match "^\s*([^#=\s]+)\s*=\s*(.*)$") {
            $key = $matches[1].Trim()
            $val = $matches[2].Trim().Trim('"').Trim("'")
            if (-not [string]::IsNullOrWhiteSpace($key)) {
                [Environment]::SetEnvironmentVariable($key, $val, "Process")
            }
        }
    }
}

# Auto-detect project if not supplied
if ([string]::IsNullOrWhiteSpace($ProjectId)) {
    $ProjectId = (gcloud config get-value project 2>$null).Trim()
    if ([string]::IsNullOrWhiteSpace($ProjectId)) {
        Write-Error "No GCP project specified and no default project configured in gcloud."
        exit 1
    }
}

$ServiceName = if ($ServiceMode -eq "streamlit") { "fraud-demo-ui" } else { "fraud-demo-api" }

Write-Host "==> Target GCP Project: $ProjectId" -ForegroundColor Cyan
Write-Host "==> Target Service: $ServiceName (mode: $ServiceMode)" -ForegroundColor Cyan
Write-Host "==> Target Region: $Region" -ForegroundColor Cyan

gcloud config set project $ProjectId

Write-Host "==> Enabling required GCP services..." -ForegroundColor Cyan
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com

$envPairs = @("SERVICE_MODE=$ServiceMode")
if ($env:LLM_PROVIDER) { $envPairs += "LLM_PROVIDER=$($env:LLM_PROVIDER)" }
if ($env:GEMINI_API_KEY) { $envPairs += "GEMINI_API_KEY=$($env:GEMINI_API_KEY)" }
if ($env:OPENAI_API_KEY) { $envPairs += "OPENAI_API_KEY=$($env:OPENAI_API_KEY)" }
$envArg = $envPairs -join ","

Write-Host "==> Submitting Cloud Build and deploying to Cloud Run..." -ForegroundColor Cyan
gcloud run deploy $ServiceName `
  --source . `
  --region $Region `
  --allow-unauthenticated `
  --memory 2Gi `
  --cpu 2 `
  --min-instances 1 `
  --set-env-vars $envArg

Write-Host "`n==> Deployment complete! Public URL:" -ForegroundColor Green
gcloud run services describe $ServiceName --region $Region --format='value(status.url)'
