param(
    [switch] $NoKitArgs,
    [string] $OmniRootName = ".omni",

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $PythonArgs
)

$ErrorActionPreference = "Stop"

$WorkspaceRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$IsaacLabRoot = "E:\IsaacLab"
$IsaacSimRoot = "E:\isaac-sim-5.1.0"
$CondaHook = Join-Path $env:USERPROFILE "miniconda3\shell\condabin\conda-hook.ps1"

if (-not (Test-Path $CondaHook)) {
    throw "Conda hook not found: $CondaHook"
}
if (-not (Test-Path $IsaacLabRoot)) {
    throw "IsaacLab root not found: $IsaacLabRoot"
}
if (-not (Test-Path $IsaacSimRoot)) {
    throw "Isaac Sim root not found: $IsaacSimRoot"
}

& $CondaHook
conda activate env_isaaclab_2.3.1

$env:ISAACLAB_ROOT = $IsaacLabRoot
$env:ISAACSIM_ROOT = $IsaacSimRoot
$env:CARB_APP_PATH = Join-Path $IsaacSimRoot "kit"
$env:EXP_PATH = Join-Path $IsaacSimRoot "apps"
$env:ISAAC_PATH = $IsaacSimRoot
$env:RESOURCE_NAME = "IsaacSim"
$env:PYTHONDONTWRITEBYTECODE = "1"

$omniRoot = Join-Path $WorkspaceRoot $OmniRootName
$env:OMNI_USER_DIR = Join-Path $omniRoot "user"
$env:OMNI_LOGS_DIR = Join-Path $omniRoot "logs"
$env:OMNI_CACHE_DIR = Join-Path $omniRoot "cache"
$env:TEMP = Join-Path $WorkspaceRoot ".tmp"
$env:TMP = $env:TEMP
$env:TMPDIR = $env:TEMP
$env:PYTHONPATH = "$IsaacSimRoot\site;$env:PYTHONPATH"

New-Item -ItemType Directory -Force -Path `
    $env:OMNI_USER_DIR, `
    $env:OMNI_LOGS_DIR, `
    $env:OMNI_CACHE_DIR, `
    $env:TEMP, `
    (Join-Path $omniRoot "portable"), `
    (Join-Path $omniRoot "data"), `
    (Join-Path $omniRoot "crash") | Out-Null

if (-not $NoKitArgs -and $PythonArgs -notcontains "--kit_args") {
    $portableRoot = (Join-Path $omniRoot "portable").Replace("\", "/")
    $kitLogFile = (Join-Path $env:OMNI_LOGS_DIR ("kit_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))).Replace("\", "/")
    $userConfigPath = (Join-Path $omniRoot "data\user.config.json").Replace("\", "/")
    $crashDir = (Join-Path $omniRoot "crash").Replace("\", "/")
    $defaultKitArgs = "--portable --portable-root=$portableRoot --/log/file=$kitLogFile --/app/userConfigPath=$userConfigPath --/crashreporter/dumpDir=$crashDir"
    $PythonArgs = @($PythonArgs + @("--kit_args", $defaultKitArgs))
}

python @PythonArgs
exit $LASTEXITCODE
