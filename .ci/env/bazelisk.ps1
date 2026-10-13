#===============================================================================
# Copyright contributors to the oneDAL project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

<#
.SYNOPSIS
Installs the pinned Windows Bazelisk executable for CI.

.DESCRIPTION
This is the PowerShell counterpart of `.ci/env/bazelisk.sh`. The Windows
nightly job cannot use that POSIX-shell helper, so this script downloads the
Windows asset, verifies the SHA256 digest published in the GitHub release, and
makes `bazel.exe` available to later workflow steps.

When GitHub Actions provides `GITHUB_PATH`, the installation directory is
appended there for subsequent steps. Otherwise, the current process `PATH` is
updated, which also makes the script convenient for local CI reproduction.
#>

$ErrorActionPreference = "Stop"

$bazeliskVersion = "v1.29.0"
$assetName = "bazelisk-windows-amd64.exe"

# SHA256 of the release assets for the version above, so the usual install path
# needs no GitHub API call: the API is the only rate-limited endpoint here, and
# it is what failed the Windows nightly on 2026-10-12 ("API rate limit exceeded
# for 130.131.238.16"). The digest is what the API would be asked for anyway, so
# pinning it loses no verification -- the download is still checked against it.
#
# Refresh together with $bazeliskVersion:
#   gh api repos/bazelbuild/bazelisk/releases/tags/<version> --jq '.assets[] | "\(.name) \(.digest)"'
#
# $pinnedVersion is what these digests belong to. Renovate bumps the version
# strings but cannot update an asset digest, so when the two differ the pins are
# ignored and the release metadata is fetched as before: a Renovate bump keeps
# working unattended, it only loses the API-free path until the digests are
# refreshed.
$pinnedVersion = "v1.29.0"
$pinnedDigests = @{
    "bazelisk-windows-amd64.exe" = "092a8738d5b41aae7a85c42cc961b1034e3389aba43ffc20c0fabda7b43e095b"
    "bazelisk-windows-arm64.exe" = "8bc42bd5d7857f18a21440b906469bb6c7cf91a7c72364d4b1e5ec56a76fe94f"
}
$installDir = Join-Path (Get-Location) "bazel\bin"
$bazelPath = Join-Path $installDir "bazel.exe"

$headers = @{
    Accept = "application/vnd.github+json"
    "X-GitHub-Api-Version" = "2022-11-28"
}

if ($env:GITHUB_TOKEN) {
    # Authentication is optional, but avoids GitHub API rate limits in CI.
    $headers.Authorization = "Bearer $env:GITHUB_TOKEN"
}

# The GitHub API and the release CDN both fail in CI without the install being
# wrong: a dropped connection, a 5xx, a secondary rate limit. Those are retried
# with exponential backoff, as `.ci/env/bazelisk.sh` does. A primary rate limit
# is reported instead of retried, see Test-RateLimitExhausted.
$fetchAttempts = if ($env:BAZELISK_FETCH_ATTEMPTS) { [int]$env:BAZELISK_FETCH_ATTEMPTS } else { 5 }
$fetchDelay = if ($env:BAZELISK_FETCH_DELAY) { [int]$env:BAZELISK_FETCH_DELAY } else { 5 }
if ($fetchAttempts -lt 1 -or $fetchDelay -lt 0) {
    throw "BAZELISK_FETCH_ATTEMPTS must be at least 1 and BAZELISK_FETCH_DELAY non-negative"
}

function Test-RateLimitExhausted {
    param([Parameter(Mandatory = $true)] $ErrorRecord)

    $response = $ErrorRecord.Exception.Response
    if (-not $response) {
        return $false
    }
    # `StatusCode` is an enum on PowerShell 7 and an int on Windows PowerShell;
    # casting handles both.
    $status = [int] $response.StatusCode
    if ($status -ne 403 -and $status -ne 429) {
        return $false
    }
    # The headers collection differs between the two editions as well, and its
    # indexer throws when the header is absent.
    $remaining = $null
    try { $remaining = $response.Headers["X-RateLimit-Remaining"] } catch { }
    if ($remaining -is [array]) { $remaining = $remaining[0] }
    return "$remaining" -eq "0"
}

function Invoke-WithRetry {
    param(
        [Parameter(Mandatory = $true)][scriptblock] $Action,
        [Parameter(Mandatory = $true)][string] $Description
    )

    $delay = $fetchDelay
    for ($attempt = 1; $attempt -le $fetchAttempts; $attempt++) {
        try {
            return & $Action
        }
        catch {
            if (Test-RateLimitExhausted -ErrorRecord $_) {
                # A primary rate limit refills only at the window reset, up to
                # an hour out, and the quota belongs to the caller, so there is
                # nothing to wait out inside the job.
                throw ("GitHub API rate limit exhausted for this runner while trying to " +
                       "$Description. Set GITHUB_TOKEN for this step to use the " +
                       "authenticated quota. Error: $($_.Exception.Message)")
            }
            if ($attempt -ge $fetchAttempts) {
                throw "Giving up after $attempt attempts: $Description. Last error: $($_.Exception.Message)"
            }
            Write-Host "Attempt $attempt of $fetchAttempts failed ($Description): $($_.Exception.Message)"
            Write-Host "Retrying in $delay s"
            Start-Sleep -Seconds $delay
            $delay = $delay * 2
        }
    }
}

$expectedHash = $null
$downloadUrl = "https://github.com/bazelbuild/bazelisk/releases/download/$bazeliskVersion/$assetName"
if ($bazeliskVersion -eq $pinnedVersion -and $pinnedDigests.ContainsKey($assetName)) {
    $expectedHash = $pinnedDigests[$assetName]
}

if (-not $expectedHash) {
    Write-Host "No pinned digest for $assetName at $bazeliskVersion; reading the release metadata."
    $release = Invoke-WithRetry -Description "fetch Bazelisk $bazeliskVersion release metadata" -Action {
        $response = Invoke-RestMethod `
            -Headers $headers `
            -Uri "https://api.github.com/repos/bazelbuild/bazelisk/releases/tags/$bazeliskVersion"
        # An empty body is a failure too, as in `.ci/env/bazelisk.sh`.
        if ($null -eq $response -or -not $response.assets) {
            throw "Empty Bazelisk release metadata response"
        }
        $response
    }

    $asset = $release.assets | Where-Object { $_.name -eq $assetName } | Select-Object -First 1
    if (-not $asset) {
        throw "Could not find $assetName in Bazelisk release $bazeliskVersion"
    }
    if (-not $asset.digest.StartsWith("sha256:")) {
        # Do not install an asset that cannot be checked against release metadata.
        throw "Unexpected Bazelisk digest format: $($asset.digest)"
    }
    $expectedHash = $asset.digest.Substring("sha256:".Length)
    $downloadUrl = $asset.browser_download_url
}

New-Item -ItemType Directory -Force -Path $installDir | Out-Null
Invoke-WithRetry -Description "download $assetName" -Action {
    Invoke-WebRequest -Uri $downloadUrl -OutFile $bazelPath
} | Out-Null

$actualHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $bazelPath).Hash.ToLowerInvariant()
if ($actualHash -ne $expectedHash.ToLowerInvariant()) {
    # Do not leave an unverified executable where the caller would add it to PATH.
    Remove-Item -LiteralPath $bazelPath -Force -ErrorAction SilentlyContinue
    throw "Bazelisk SHA256 mismatch. Expected $expectedHash, got $actualHash"
}

$pathLine = $installDir
if ($env:GITHUB_PATH) {
    $pathLine | Out-File -FilePath $env:GITHUB_PATH -Encoding utf8 -Append
}
else {
    $env:PATH = "$installDir;$env:PATH"
}

& $bazelPath --version
