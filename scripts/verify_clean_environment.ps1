param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [string]$VenvName = "clean-base-verify-venv",
    [switch]$ResumeExisting
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:NO_COLOR = "1"
$env:PIP_DISABLE_PIP_VERSION_CHECK = "1"
$env:PIP_DEFAULT_TIMEOUT = "120"
$env:PYTHONPATH = $null
$env:VIPIBENCH_CLEAN_ENV_BOOTSTRAP = "1"

$venv = Join-Path $ProjectRoot ("build\" + $VenvName)
if (Test-Path -LiteralPath $venv) {
    if (-not $ResumeExisting) {
        throw "Refusing to reuse an existing clean verification environment: $venv"
    }
} else {
    & py -3.11 -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed" }
}

$python = Join-Path $venv "Scripts\python.exe"
$env:HF_HOME = Join-Path $venv "hf-home"
$testReport = Join-Path $ProjectRoot ("build\" + $VenvName + "-tests.xml")
$dependencyLock = Join-Path $ProjectRoot "requirements-experiment.lock"
$commands = @()
if (-not $ResumeExisting) {
    $commands += ,@("-m", "pip", "install", "--no-color", "--progress-bar", "off", "--retries", "8", "-r", $dependencyLock)
    $commands += ,@("-m", "pip", "install", "--no-color", "--progress-bar", "off", "--retries", "8", "--no-deps", $ProjectRoot)
} else {
    $commands += ,@("-m", "pip", "install", "--no-color", "--progress-bar", "off", "--retries", "8", "-r", $dependencyLock)
    $commands += ,@("-m", "pip", "install", "--no-color", "--progress-bar", "off", "--retries", "8", "--no-deps", "--force-reinstall", $ProjectRoot)
}
$commands += ,@("-m", "pip", "check")
$commands += ,@("-m", "vipibench.cli", "doctor")
$commands += ,@("-m", "vipibench.cli", "verify-environment-compatibility", "--project-root", $ProjectRoot, "--output", (Join-Path $ProjectRoot "outputs\environment_compatibility.json"))
$commands += ,@("-m", "pytest", "-q", "-o", "pythonpath=", "--junitxml", $testReport)

$results = @()
foreach ($arguments in $commands) {
    & $python @arguments
    $exitCode = $LASTEXITCODE
    $results += [ordered]@{ arguments = $arguments; exit_code = $exitCode }
    if ($exitCode -ne 0) { throw "Clean-environment gate failed: $($arguments -join ' ')" }
}
$installedModule = & $python -c "from pathlib import Path; import vipibench; print(Path(vipibench.__file__).resolve())"
if ($LASTEXITCODE -ne 0) { throw "installed module path check failed" }
$sourcePackage = (Resolve-Path -LiteralPath (Join-Path $ProjectRoot "src\vipibench")).Path.TrimEnd("\") + "\"
if ($installedModule.Trim().StartsWith($sourcePackage, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Clean-environment verification imported source instead of the installed project: $installedModule"
}
$fingerprint = & $python -c "from pathlib import Path; from vipibench.manifest import runtime_source_fingerprint; print(runtime_source_fingerprint(Path.cwd()))"
if ($LASTEXITCODE -ne 0) { throw "runtime fingerprint failed" }
[xml]$testXml = Get-Content -LiteralPath $testReport -Raw
$testCount = [int]$testXml.testsuites.testsuite.tests
if ($testCount -le 0) { throw "JUnit report did not contain a positive test count" }

$artifact = [ordered]@{
    schema_version = "1.0.0"
    status = "PASS"
    python = $python
    isolated_venv = $venv
    resumed_after_interrupted_bootstrap = [bool]$ResumeExisting
    commands = $results
    runtime_source_fingerprint = $fingerprint.Trim()
    dependency_lock_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $dependencyLock).Hash
    test_count = $testCount
    imported_module_path = $installedModule.Trim()
    note = "The isolated environment is generated under ignored build/ and is not a release artifact."
}
$output = Join-Path $ProjectRoot "outputs\clean_environment_verification.json"
$json = $artifact | ConvertTo-Json -Depth 8
[System.IO.File]::WriteAllText($output, $json, (New-Object System.Text.UTF8Encoding($false)))
Write-Output "Clean environment verification PASS: $output"
