<#
.SYNOPSIS
  Scan an entire S3 bucket and list every object whose
  x-amz-meta-human_verification metadata is true,
  showing ✓ or ✗ per file and a final tally.

.PARAMETER Bucket
  (Required) The S3 bucket name to scan.

.PARAMETER Region
  (Optional) AWS region (default "eu-north-1").
#>
Param(
  [Parameter(Mandatory=$true)][string]$Bucket,
  [string]$Region = 'eu-north-1'
)

Write-Host "Listing objects in bucket '$Bucket' region '$Region'..."
# 1) Grab all keys (aws CLI auto-paginates)
$keysJson = aws s3api list-objects-v2 `
  --bucket $Bucket `
  --region $Region `
  --query 'Contents[].Key' `
  --output json
if ($LASTEXITCODE -ne 0) {
  Write-Error "Failed to list objects; check bucket name/region/credentials."
  exit 1
}

$keys = $keysJson | ConvertFrom-Json
if (-not $keys -or $keys.Count -eq 0) {
  Write-Host 'No objects found in that bucket.'
  exit 0
}

# 2) Check each key’s metadata
$matches = @()
foreach ($key in $keys) {
  Write-Host "Checking metadata for: $key"

  $val = aws s3api head-object `
    --bucket $Bucket `
    --key   $key `
    --region $Region `
    --query 'Metadata.human_verification' `
    --output text 2>$null

  if ($LASTEXITCODE -ne 0) {
    Write-Warning "  ↳ Could not fetch metadata for $key"
    continue
  }

  $trim = $val.Trim()
  if ($trim -eq 'true') {
    Write-Host '  human_verification=true' -ForegroundColor Green
    $matches += $key
  }
  else {
    Write-Host "  human_verification!=true (got '$trim')"
  }
}

# 3) Final summary
if ($matches.Count -gt 0) {
  Write-Host ""
  Write-Host "Found $($matches.Count) object(s) with human_verification=true:"
  $matches | ForEach-Object { Write-Host " - $_" }
}
else {
  Write-Host 'No objects found with human_verification=true.'
}
