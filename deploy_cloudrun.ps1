# Deploy the fraud-detection engine to Google Cloud Run from PowerShell
param (
    [Parameter(Mandatory=$true, Position=0)]
    [string]$ProjectId,

    [Parameter(Position=1)]
    [string]$Region = "us-central1",

    [Parameter(Position=2)]
    [ValidateSet("api", "streamlit")]
    [string]$ServiceMode = "api"
)

$Service = "fraud-engine"

Write-Host "==> Setting active GCP project to $ProjectId..." -ForegroundColor Cyan
gcloud config set project $ProjectId

Write-Host "==> Enabling required GCP services..." -ForegroundColor Cyan
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com

$envVars = "SERVICE_MODE=$ServiceMode"
if ($env:LLM_PROVIDER) { $envVars += ",LLM_PROVIDER=$($env:LLM_PROVIDER)" }
if ($env:GEMINI_API_KEY) { $envVars += ",GEMINI_API_KEY=$($env:GEMINI_API_KEY)" }
if ($env:OPENAI_API_KEY) { $envVars += ",OPENAI_API_KEY=$($env:OPENAI_API_KEY)" }

Write-Host "==> Deploying $Service (mode: $ServiceMode) to $Region via Cloud Build..." -ForegroundColor Cyan
gcloud run deploy $Service `
  --source . `
  --region $Region `
  --allow-unauthenticated `
  --memory 2Gi `
  --cpu 2 `
  --min-instances 1 `
  --set-env-vars $envVars

Write-Host "==> Deployment complete! Service URL:" -ForegroundColor Green
gcloud run services describe $Service --region $Region --format='value(status.url)'
