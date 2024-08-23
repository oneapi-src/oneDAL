@echo off
rem ============================================================================
rem Copyright 2018 Intel Corporation
rem
rem Licensed under the Apache License, Version 2.0 (the "License");
rem you may not use this file except in compliance with the License.
rem You may obtain a copy of the License at
rem
rem     http://www.apache.org/licenses/LICENSE-2.0
rem
rem Unless required by applicable law or agreed to in writing, software
rem distributed under the License is distributed on an "AS IS" BASIS,
rem WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
rem See the License for the specific language governing permissions and
rem limitations under the License.
rem ============================================================================

rem req: PowerShell 3.0+
powershell.exe -command "if ($PSVersionTable.PSVersion.Major -ge 3) {exit 1} else {Write-Host \"The script requires PowerShell 3.0 or above (current version: $($PSVersionTable.PSVersion.Major).$($PSVersionTable.PSVersion.Minor))\"}" && goto Error_load

set MKLURLROOT=https://registrationcenter-download.intel.com/akdlm/irc_nas/
set MKLVERSION=2024.2.0

set MKLPACKAGE=l_oneapi_mkl_p_%MKLVERSION%_offline

set MKLURL=%MKLURLROOT%%MKLPACKAGE%.zip

if /i "%1"=="" (
    set DST=%~dp0..\__deps\oneMKL
) else (
    set DST=%1\..\__deps\oneMKL
)

CALL :Download_MKL %DST% , %MKLURL% , %MKLPACKAGE%

exit /B 0

:Download_MKL

set DST=%~1
set SRC=%~2
set FILENAME=%~3

if not exist %DST% mkdir %DST%

if not exist "%DST%\lib" (
    powershell.exe -command "(New-Object System.Net.WebClient).DownloadFile('%SRC%', '%DST%\%FILENAME%.zip')" && goto Unpack || goto Error_load

:Unpack
    powershell.exe -command "if (Get-Command Add-Type -errorAction SilentlyContinue) {Add-Type -Assembly \"System.IO.Compression.FileSystem\"; try { [IO.Compression.zipfile]::ExtractToDirectory(\"%DST%\%FILENAME%.zip\", \"%DST%\")}catch{$_.exception ; exit 1}} else {exit 1}" && goto Exit || goto Error_unpack

:Error_load
    echo download_onemkl.bat : Error: Failed to load %SRC% to %DST%, try to load it manually
    exit /B 1

:Error_unpack
    echo download_onemkl.bat : Error: Failed to unpack %DST%\%FILENAME%.zip to %DST%, try unpack the archive manually
    exit /B 1

:Exit
    echo Downloaded and unpacked Intel^(R^) oneMKL libraries to %DST%
    exit /B 0
) else (
    echo Intel^(R^) oneMKL libraries are already installed in %DST%
    exit /B 0
)
