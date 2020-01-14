@echo off
rem ============================================================================
rem Copyright 2014-2020 Intel Corporation
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

setlocal
call:GetFullPath "%~dp0.."    DAAL
call:GetFullPath "%~dp0..\.." DAAL_UP

:ParseArgs
if /i "%1"=="" goto :EndParseArgs
if /i "%1"=="ia32"    (set DAAL_IA=ia32)    & shift & goto :ParseArgs
if /i "%1"=="intel64" (set DAAL_IA=intel64) & shift & goto :ParseArgs
if /i "%1"=="vs2013"                          shift & goto :ParseArgs
if /i "%1"=="vs2015"                          shift & goto :ParseArgs
if /i "%1"=="vs2017"                          shift & goto :ParseArgs
if /i "%1"=="vs2019"                          shift & goto :ParseArgs
goto :Usage

:EndParseArgs
if /i not "%DAAL_IA%"=="" goto :GoodArgs
goto :Usage

:GetFullPath
set %2=%~f1
exit /b 0

:Usage
echo.
echo Syntax:  call %~nx0 ^<arch^>
echo Where ^<arch^> is one of
echo   ia32     - setup environment for IA-32 architecture
echo   intel64  - setup environment for Intel(R) 64 architecture
exit /b 1

:GoodArgs
set "DAALROOT=%DAAL%"
set "INCLUDE=%DAAL%\include;%INCLUDE%"
set "CPATH=%DAAL%\include;%CPATH%"
if not defined TBBROOT (
    set "LIB=%DAAL%\lib\%DAAL_IA%_win;%DAAL_UP%\tbb\lib\%DAAL_IA%_win\vc_mt;%LIB%"
    set "PATH=%DAAL_UP%\redist\%DAAL_IA%_win\daal;%DAAL_UP%\redist\%DAAL_IA%_win\tbb\vc_mt;%PATH%"
    if exist "%DAAL_UP%\..\linux\daal\lib\intel64_lin" (
        set "LD_LIBRARY_PATH=%DAAL_UP%\..\linux\daal\lib\intel64_lin;%DAAL_UP%\..\linux\tbb\lib\intel64_lin\gcc4.7;%DAAL_UP%\..\linux\tbb\lib\intel64_lin\gcc4.4;%LD_LIBRARY_PATH%"
    )
) else (
    set "LIB=%DAAL%\lib\%DAAL_IA%_win;%LIB%"
    set "PATH=%DAAL_UP%\redist\%DAAL_IA%_win\daal;%PATH%"
    if exist "%DAAL_UP%\..\linux\daal\lib\intel64_lin" (
        set "LD_LIBRARY_PATH=%DAAL_UP%\..\linux\daal\lib\intel64_lin;%LD_LIBRARY_PATH%"
    )
)
set "CLASSPATH=%DAAL%\lib\daal.jar;%CLASSPATH%"
endlocal& ^
set DAALROOT=%DAALROOT%& ^
set INCLUDE=%INCLUDE%& ^
set CPATH=%CPATH%& ^
set LIB=%LIB%& ^
set PATH=%PATH%& ^
set LD_LIBRARY_PATH=%LD_LIBRARY_PATH%& ^
set CLASSPATH=%CLASSPATH%& ^
goto:eof
