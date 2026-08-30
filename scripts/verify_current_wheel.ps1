param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [string]$VenvName = "clean-base-verify-venv"
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:NO_COLOR = "1"
$env:PIP_DEFAULT_TIMEOUT = "120"
$env:VIPIBENCH_CURRENT_WHEEL_BOOTSTRAP = "1"
$env:PYTHONPATH = $null

$venv = Join-Path $ProjectRoot ("build\" + $VenvName)
$python = Join-Path $venv "Scripts\python.exe"
$env:HF_HOME = Join-Path $venv "hf-home"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Run verify_clean_environment.ps1 first"
}

$wheelDir = Join-Path $ProjectRoot "build\current-wheel"
if (Test-Path -LiteralPath $wheelDir) {
    $resolvedRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path.TrimEnd("\") + "\"
    $resolvedWheelDir = (Resolve-Path -LiteralPath $wheelDir).Path
    if (-not $resolvedWheelDir.StartsWith($resolvedRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Unsafe wheel directory: $resolvedWheelDir"
    }
    Remove-Item -LiteralPath $resolvedWheelDir -Recurse -Force
}
New-Item -ItemType Directory -Path $wheelDir | Out-Null

& $python -m pip wheel --no-color --progress-bar off --retries 8 --no-deps --wheel-dir $wheelDir $ProjectRoot
if ($LASTEXITCODE -ne 0) { throw "Current wheel build failed" }
$wheels = @(Get-ChildItem -LiteralPath $wheelDir -Filter "vipibench_guardrail-*.whl")
if ($wheels.Count -ne 1) { throw "Expected exactly one project wheel; found $($wheels.Count)" }
$wheel = $wheels[0]
& $python -m pip install --no-color --progress-bar off --no-deps --force-reinstall $wheel.FullName
if ($LASTEXITCODE -ne 0) { throw "Current wheel install failed" }
& $python -m pip check
if ($LASTEXITCODE -ne 0) { throw "pip check failed" }
$installedModule = & $python -c "from pathlib import Path; import vipibench; print(Path(vipibench.__file__).resolve())"
if ($LASTEXITCODE -ne 0) { throw "installed module path check failed" }
$sourcePackage = (Resolve-Path -LiteralPath (Join-Path $ProjectRoot "src\vipibench")).Path.TrimEnd("\") + "\"
if ($installedModule.Trim().StartsWith($sourcePackage, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Current-wheel verification imported source instead of the installed wheel: $installedModule"
}
& $python -m vipibench.cli doctor
if ($LASTEXITCODE -ne 0) { throw "doctor failed" }
& $python -m vipibench.cli verify-environment-compatibility --project-root $ProjectRoot --output (Join-Path $ProjectRoot "outputs\environment_compatibility.json")
if ($LASTEXITCODE -ne 0) { throw "environment compatibility failed" }
Push-Location $ProjectRoot
try {
    $testReport = Join-Path $ProjectRoot "build\current-wheel-tests.xml"
    & $python -m pytest -q -o "pythonpath=" --junitxml $testReport
    if ($LASTEXITCODE -ne 0) { throw "tests failed" }
    [xml]$testXml = Get-Content -LiteralPath $testReport -Raw
    $testCount = [int]$testXml.testsuites.testsuite.tests
    if ($testCount -le 0) { throw "JUnit report did not contain a positive test count" }
    $fingerprint = & $python -c "from pathlib import Path; from vipibench.manifest import runtime_source_fingerprint; print(runtime_source_fingerprint(Path.cwd()))"
} finally {
    Pop-Location
}

$artifact = [ordered]@{
    schema_version = "1.0.0"
    status = "PASS"
    wheel_path = $wheel.FullName
    wheel_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $wheel.FullName).Hash
    runtime_source_fingerprint = $fingerprint.Trim()
    requirements_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $ProjectRoot "requirements.lock")).Hash
    experiment_lock_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $ProjectRoot "requirements-experiment.lock")).Hash
    test_count = $testCount
    imported_module_path = $installedModule.Trim()
    note = "Current runtime source was packaged as a wheel, force-installed into the clean dependency environment, and re-verified."
}
$output = Join-Path $ProjectRoot "outputs\current_wheel_verification.json"
$json = $artifact | ConvertTo-Json -Depth 5
[System.IO.File]::WriteAllText($output, $json, (New-Object System.Text.UTF8Encoding($false)))
Write-Output "Current wheel verification PASS: $output"
