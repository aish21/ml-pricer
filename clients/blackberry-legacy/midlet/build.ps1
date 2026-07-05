param(
    [string]$WTK_HOME = $env:WTK_HOME,
    [string]$JavaHome = $env:JAVA_HOME,
    [string]$JavacSource = $(if ($env:JAVAC_SOURCE) { $env:JAVAC_SOURCE } else { "1.3" }),
    [string]$JavacTarget = $(if ($env:JAVAC_TARGET) { $env:JAVAC_TARGET } else { "1.1" })
)

$ErrorActionPreference = "Stop"

function Fail($message) {
    Write-Error $message
    exit 1
}

function Resolve-Tool($name, $candidatePaths) {
    foreach ($path in $candidatePaths) {
        if ($path -and (Test-Path -LiteralPath $path)) {
            return (Resolve-Path -LiteralPath $path).Path
        }
    }

    $command = Get-Command $name -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }

    return $null
}

function Resolve-FirstExisting($candidatePaths, $label) {
    foreach ($path in $candidatePaths) {
        if ($path -and (Test-Path -LiteralPath $path)) {
            return (Resolve-Path -LiteralPath $path).Path
        }
    }
    Fail "Could not find $label. Check WTK_HOME and the Java ME SDK installation."
}

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$SourceDir = Join-Path $ProjectRoot "src"
$ManifestPath = Join-Path $ProjectRoot "manifest\MANIFEST.MF"
$TemplatePath = Join-Path $ProjectRoot "AshBerryTerminal.jad.template"
$BuildDir = Join-Path $ProjectRoot "build"
$ClassesDir = Join-Path $BuildDir "classes"
$PreverifiedDir = Join-Path $BuildDir "preverified"
$DistDir = Join-Path $ProjectRoot "dist"
$JarPath = Join-Path $DistDir "AshBerryTerminal.jar"
$JadPath = Join-Path $DistDir "AshBerryTerminal.jad"

if (-not $WTK_HOME) {
    Fail "WTK_HOME is not set. Install Sun Java Wireless Toolkit 2.5.2_01 or a compatible Java ME SDK, then set WTK_HOME, for example: `$env:WTK_HOME='C:\WTK252'"
}
if (-not (Test-Path -LiteralPath $WTK_HOME)) {
    Fail "WTK_HOME does not exist: $WTK_HOME"
}

$JavacPath = Resolve-Tool "javac" @(
    $(if ($JavaHome) { Join-Path $JavaHome "bin\javac.exe" } else { $null })
)
$JarToolPath = Resolve-Tool "jar" @(
    $(if ($JavaHome) { Join-Path $JavaHome "bin\jar.exe" } else { $null })
)
$PreverifyPath = Resolve-Tool "preverify" @(
    (Join-Path $WTK_HOME "bin\preverify.exe"),
    (Join-Path $WTK_HOME "bin\preverifier.exe")
)
if (-not $PreverifyPath) {
    $PreverifyPath = Resolve-Tool "preverifier" @(
        (Join-Path $WTK_HOME "bin\preverifier.exe"),
        (Join-Path $WTK_HOME "bin\preverify.exe")
    )
}

if (-not $JavacPath) {
    Fail "javac was not found. Set JAVA_HOME or add javac to PATH."
}
if (-not $JarToolPath) {
    Fail "jar was not found. Set JAVA_HOME or add jar to PATH."
}
if (-not $PreverifyPath) {
    Fail "preverify/preverifier was not found. Expected it under WTK_HOME\bin or PATH."
}

$CldcJar = Resolve-FirstExisting @(
    (Join-Path $WTK_HOME "lib\cldcapi11.jar"),
    (Join-Path $WTK_HOME "lib\cldcapi10.jar"),
    (Join-Path $WTK_HOME "lib\cldc_1.1.jar"),
    (Join-Path $WTK_HOME "lib\cldc-1.1.jar")
) "CLDC API jar"
$MidpJar = Resolve-FirstExisting @(
    (Join-Path $WTK_HOME "lib\midpapi20.jar"),
    (Join-Path $WTK_HOME "lib\midpapi21.jar"),
    (Join-Path $WTK_HOME "lib\midp_2.0.jar"),
    (Join-Path $WTK_HOME "lib\midp-2.0.jar")
) "MIDP API jar"

$ClassPath = "$CldcJar;$MidpJar"
$Sources = Get-ChildItem -Path $SourceDir -Recurse -Filter "*.java" | ForEach-Object { $_.FullName }
if (-not $Sources) {
    Fail "No Java source files found under $SourceDir"
}

Remove-Item -LiteralPath $BuildDir -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $DistDir -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $ClassesDir, $PreverifiedDir, $DistDir | Out-Null

Write-Host "Compiling MIDlet..."
Write-Host "  javac: $JavacPath"
Write-Host "  source/target: $JavacSource / $JavacTarget"
& $JavacPath `
    -bootclasspath $ClassPath `
    -source $JavacSource `
    -target $JavacTarget `
    -d $ClassesDir `
    $Sources

Write-Host "Preverifying classes..."
& $PreverifyPath `
    -classpath "$ClassPath;$ClassesDir" `
    -d $PreverifiedDir `
    $ClassesDir

Write-Host "Packaging JAR..."
& $JarToolPath cfm $JarPath $ManifestPath -C $PreverifiedDir .

$JarSize = (Get-Item -LiteralPath $JarPath).Length
(Get-Content -LiteralPath $TemplatePath -Raw).Replace("UPDATE_AFTER_BUILD", [string]$JarSize) |
    Set-Content -LiteralPath $JadPath -Encoding ASCII

Write-Host ""
Write-Host "Build complete."
Write-Host "  JAR: $JarPath"
Write-Host "  JAD: $JadPath"
