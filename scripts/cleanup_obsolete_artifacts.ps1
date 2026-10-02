param(
    [switch]$Execute,
    [string]$BackupDirectory,
    [switch]$IncludeLegacyHelpers,
    # Compatibility only: archived experiments are always protected now.
    [switch]$KeepArchivedExperiments
)

$ErrorActionPreference = 'Stop'
$workspace = [System.IO.Path]::GetFullPath((Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).ProviderPath)
$separator = [System.IO.Path]::DirectorySeparatorChar
$protectedRoots = @('outputs', 'data', '.git', '.codex', '.agents', 'archive', '.venv', 'venv', 'env')
$cacheNames = @('__pycache__', '.pytest_cache', '.mypy_cache', '.ruff_cache')
$artifactExtensions = @('.xlsx', '.xls', '.xlsm', '.csv', '.parquet', '.pt', '.pth', '.ckpt', '.onnx', '.png', '.pdf', '.svg', '.docx', '.bkp', '.apw', '.apwz', '.asp')

function Assert-WorkspaceChild([string]$Path) {
    $resolved = [System.IO.Path]::GetFullPath($Path)
    if (-not $resolved.StartsWith($workspace + $separator, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing path outside workspace: $resolved"
    }
    foreach ($rootName in $protectedRoots) {
        $root = Join-Path $workspace $rootName
        if ($resolved.Equals($root, [System.StringComparison]::OrdinalIgnoreCase) -or
            $resolved.StartsWith($root + $separator, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "Refusing protected path: $resolved"
        }
    }
    return $resolved
}

$targets = [System.Collections.Generic.List[object]]::new()
function Add-CleanupTarget([string]$Path, [string]$Reason) {
    if (Test-Path -LiteralPath $Path) {
        $resolved = Assert-WorkspaceChild $Path
        if ($targets.Path -notcontains $resolved) {
            $targets.Add([pscustomobject]@{ Path = $resolved; Reason = $Reason })
        }
    }
}

# Traverse only source trees, never the data or output archive.
foreach ($cacheName in $cacheNames) {
    Add-CleanupTarget (Join-Path $workspace $cacheName) 'Disposable tooling cache'
}
foreach ($sourceRoot in @('src', 'tests', 'scripts', 'economic module GA')) {
    $start = Join-Path $workspace $sourceRoot
    if (-not (Test-Path -LiteralPath $start -PathType Container)) { continue }
    $queue = [System.Collections.Generic.Queue[string]]::new()
    $queue.Enqueue($start)
    while ($queue.Count -gt 0) {
        foreach ($dir in Get-ChildItem -LiteralPath ($queue.Dequeue()) -Directory -Force) {
            # Reparse points are not followed; protected subtrees are not scanned.
            if (($dir.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0 -or
                $dir.Name -in ($protectedRoots + @('logs', 'runs', 'wandb'))) { continue }
            if ($dir.Name -in $cacheNames) {
                Add-CleanupTarget $dir.FullName 'Disposable Python/tooling cache'
            } else {
                $queue.Enqueue($dir.FullName)
            }
        }
    }
}

$emptyRoot = Join-Path $workspace 'a5000v2'
if ((Test-Path -LiteralPath $emptyRoot -PathType Container) -and
    @(Get-ChildItem -LiteralPath $emptyRoot -Force).Count -eq 0) {
    Add-CleanupTarget $emptyRoot 'Empty temporary directory'
}

if ($IncludeLegacyHelpers) {
    foreach ($relative in @(
        'scripts/run_0727_pinnsafe_single_pipeline.sh',
        'scripts/run_factorial_260716_batch1.sh',
        'scripts/run_factorial_260716_batch2.sh',
        'scripts/wait_and_audit_targetrow_debug.ps1',
        'docs/model_FINAL0818.md'
    )) {
        Add-CleanupTarget (Join-Path $workspace $relative) 'Retired unreferenced helper/draft; recoverable archive'
    }
    # Identify the reviewed pasted-math draft by content, avoiding locale-dependent filenames.
    foreach ($draft in Get-ChildItem -LiteralPath (Join-Path $workspace 'docs') -Filter '*.md' -File) {
        if ((Get-FileHash -LiteralPath $draft.FullName).Hash -eq
            '0D9E8377443EC920E5D0E50F85C6C72A7A2A8FFBFA5B4425615F7A1D23B45696') {
            Add-CleanupTarget $draft.FullName 'Unreferenced pasted-math draft; recoverable archive'
        }
    }
    $duplicate = Join-Path $workspace 'docs/final_model.md'
    $canonical = Join-Path $workspace 'docs/MODEL_final.md'
    if ((Test-Path -LiteralPath $duplicate) -and (Test-Path -LiteralPath $canonical) -and
        (Get-FileHash -LiteralPath $duplicate).Hash -eq (Get-FileHash -LiteralPath $canonical).Hash) {
        Add-CleanupTarget $duplicate 'Exact duplicate of docs/MODEL_final.md'
    }
    # A known invalid patch mirror, not the real server root. If it changes, leave it alone.
    $mirror = Join-Path $workspace 'raidrive-researcher'
    if (Test-Path -LiteralPath $mirror -PathType Container) {
        $mirrorFiles = @(Get-ChildItem -LiteralPath $mirror -Recurse -File -Force)
        if ($mirrorFiles.Count -eq 1 -and $mirrorFiles[0].Name -eq 'rename_final2_model_labels.py') {
            Add-CleanupTarget $mirror 'Invalid one-file path mirror; authoritative script retained'
        }
    }
}

$rows = @()
$fileRows = @()
foreach ($target in $targets | Sort-Object Path) {
    $resolved = Assert-WorkspaceChild $target.Path
    $item = Get-Item -LiteralPath $resolved -Force
    if (($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Refusing reparse-point target: $resolved"
    }
    $files = if ($item.PSIsContainer) { @(Get-ChildItem -LiteralPath $resolved -Recurse -File -Force) } else { @($item) }
    foreach ($file in $files) {
        $null = Assert-WorkspaceChild $file.FullName
        if ($file.Extension.ToLowerInvariant() -in $artifactExtensions) {
            throw "Refusing to move a research artifact: $($file.FullName)"
        }
        $fileRows += [pscustomobject]@{
            RelativePath = $file.FullName.Substring($workspace.Length + 1)
            Bytes = $file.Length
            SHA256 = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash
        }
    }
    $rows += [pscustomobject]@{
        RelativePath = $resolved.Substring($workspace.Length + 1)
        ItemKind = $(if ($item.PSIsContainer) { 'Directory' } else { 'File' })
        FileCount = $files.Count
        Bytes = [long](($files | Measure-Object Length -Sum).Sum)
        Reason = $target.Reason
    }
}

$rows | Format-Table RelativePath, FileCount, Bytes, Reason -AutoSize
Write-Host ("Candidates: {0} paths, {1} files. ALL data, outputs, and research artifacts are protected." -f $rows.Count, $fileRows.Count)
if (-not $Execute) {
    Write-Host 'Preview only. Use -Execute to move candidates into a recoverable sibling backup.'
    return
}
if ($rows.Count -eq 0) { Write-Host 'Nothing to clean.'; return }

if (-not $BackupDirectory) {
    $BackupDirectory = Join-Path (Split-Path -Parent $workspace) ('_ProcGNN_cleanup_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
}
$backup = [System.IO.Path]::GetFullPath($BackupDirectory)
if ($backup.Equals($workspace, [System.StringComparison]::OrdinalIgnoreCase) -or
    $backup.StartsWith($workspace + $separator, [System.StringComparison]::OrdinalIgnoreCase) -or
    -not (Split-Path -Leaf $backup).StartsWith('_ProcGNN_', [System.StringComparison]::OrdinalIgnoreCase)) {
    throw 'Backup must be outside the workspace and its directory name must start with _ProcGNN_.'
}
if (Test-Path -LiteralPath $backup) { throw "Backup directory already exists: $backup" }
New-Item -ItemType Directory -Path $backup | Out-Null
$rows | Export-Csv -LiteralPath (Join-Path $backup 'cleanup_manifest.csv') -NoTypeInformation -Encoding UTF8
$fileRows | Export-Csv -LiteralPath (Join-Path $backup 'file_checksums.csv') -NoTypeInformation -Encoding UTF8
$rows | ForEach-Object { $_ | Add-Member -NotePropertyName OriginalWorkspace -NotePropertyValue $workspace }
$rows | Export-Csv -LiteralPath (Join-Path $backup 'restore_manifest.csv') -NoTypeInformation -Encoding UTF8

foreach ($row in $rows) {
    $source = Assert-WorkspaceChild (Join-Path $workspace $row.RelativePath)
    $destination = [System.IO.Path]::GetFullPath((Join-Path (Join-Path $backup 'files') $row.RelativePath))
    if (-not $destination.StartsWith($backup + $separator, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing destination outside backup: $destination"
    }
    New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
    Move-Item -LiteralPath $source -Destination $destination
    Write-Host "Archived: $($row.RelativePath)"
}
Write-Host "Cleanup complete; nothing permanently deleted. Backup: $backup"
Write-Host 'Restore: use restore_manifest.csv to copy each files/<RelativePath> back to OriginalWorkspace/<RelativePath>.'
