<#!
.SYNOPSIS
Runs the repeatable, constrained GNN-economic optimization study for the paper.

.DESCRIPTION
Each arm uses the same operating bounds, H2-throughput constraint, process
templates, and random seeds for GA and the equal-budget random-search control.
The script intentionally never enables --unsafe-input-proxies.
#>
[CmdletBinding()]
param(
    [ValidateSet('smoke', 'pilot', 'paper')]
    [string]$Profile = 'pilot',

    [ValidateSet('primary', 'co2', 'all')]
    [string]$Arms = 'primary',

    [string]$Processes = 'all',

    [ValidateSet('ga', 'random')]
    [string[]]$Methods = @('ga', 'random'),

    [ValidateSet('auto', 'cpu', 'cuda')]
    [string]$Device = 'auto',

    [string]$OutputRoot = ''
)

$ErrorActionPreference = 'Stop'
$moduleDir = $PSScriptRoot
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $moduleDir '..\..'))
$python = (Get-Command python -ErrorAction Stop).Source

switch ($Profile) {
    'smoke' { $seeds = @(101, 202); $population = 4;  $generations = 1 }
    'pilot' { $seeds = @(101, 202, 303); $population = 12; $generations = 8 }
    'paper' { $seeds = @(101, 202, 303, 404, 505); $population = 24; $generations = 30 }
}

$scenarios = @()
if ($Arms -in @('primary', 'all')) {
    $scenarios += [PSCustomObject]@{ Name = 'LCOH_capacity'; Co2Fraction = $null }
}
if ($Arms -in @('co2', 'all')) {
    # A relative cap is used so that every process is compared against its own
    # GNN template instead of imposing an arbitrary cross-process CO2 number.
    $scenarios += [PSCustomObject]@{ Name = 'CO2_90pct'; Co2Fraction = 0.90 }
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    $OutputRoot = Join-Path $repoRoot "outputs\paper_economic_ga_$stamp"
}
$OutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)
if (Test-Path -LiteralPath $OutputRoot) {
    throw "OutputRoot already exists: $OutputRoot. Use a new directory to prevent mixing study arms."
}
New-Item -ItemType Directory -Path $OutputRoot | Out-Null

$processCount = if ($Processes.Trim().ToLowerInvariant() -eq 'all') { 10 } else { @($Processes -split ',' | Where-Object { $_.Trim() }).Count }
$evaluationsPerProcessRun = $population * ($generations + 1)
$totalEvaluations = $processCount * $evaluationsPerProcessRun * $seeds.Count * $Methods.Count * $scenarios.Count
Write-Host "[STUDY] profile=$Profile arms=$Arms methods=$($Methods -join ',')"
Write-Host "[STUDY] root=$OutputRoot"
Write-Host "[STUDY] planned GNN-economic evaluations=$totalEvaluations ($evaluationsPerProcessRun per process/run)"

Push-Location $moduleDir
try {
    foreach ($scenario in $scenarios) {
        foreach ($method in $Methods) {
            foreach ($seed in $seeds) {
                $runDir = Join-Path $OutputRoot (Join-Path $scenario.Name (Join-Path $method "seed_$seed"))
                $runArgs = @(
                    'run_ga_optimization.py',
                    '--processes', $Processes,
                    '--search-method', $method,
                    '--population', $population,
                    '--generations', $generations,
                    '--seed', $seed,
                    '--output-dir', $runDir
                )
                if ($null -ne $scenario.Co2Fraction) {
                    $runArgs += @('--co2-max-fraction', $scenario.Co2Fraction)
                }
                if ($Device -ne 'auto') {
                    $runArgs += @('--device', $Device)
                }
                Write-Host "[STUDY] scenario=$($scenario.Name) method=$method seed=$seed"
                & $python @runArgs
                if ($LASTEXITCODE -ne 0) {
                    throw "Optimization failed: scenario=$($scenario.Name), method=$method, seed=$seed"
                }
            }
        }
    }
    & $python 'summarize_paper_ga.py' '--root' $OutputRoot
    if ($LASTEXITCODE -ne 0) {
        throw "Study aggregation failed for $OutputRoot"
    }
}
finally {
    Pop-Location
}

Write-Host "[STUDY] complete: $OutputRoot"
