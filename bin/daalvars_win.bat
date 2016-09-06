@echo off
rem ============================================================================
rem Copyright 2014-2016 Intel Corporation
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
if /i "%1"=="none"                            shift & goto :ParseArgs
if /i "%1"=="vs2012"                          shift & goto :ParseArgs
if /i "%1"=="vs2013"                          shift & goto :ParseArgs
if /i "%1"=="vs2015"                          shift & goto :ParseArgs
if /i "%1"=="vs2010"                          shift & goto :ParseArgs
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
if not defined TBBROOT (
    set "LIB=%DAAL%\lib\%DAAL_IA%_win;%DAAL_UP%\tbb\lib\%DAAL_IA%_win\vc_mt;%LIB%"
    set "PATH=%DAAL_UP%\redist\%DAAL_IA%_win\daal;%DAAL_UP%\redist\%DAAL_IA%_win\tbb\vc_mt;%PATH%"
) else (
    set "LIB=%DAAL%\lib\%DAAL_IA%_win;%LIB%"
    set "PATH=%DAAL_UP%\redist\%DAAL_IA%_win\daal;%PATH%"
)
set "CLASSPATH=%DAAL%\lib\daal.jar;%CLASSPATH%"
endlocal& ^
set DAALROOT=%DAALROOT%& ^
set INCLUDE=%INCLUDE%& ^
set LIB=%LIB%& ^
set PATH=%PATH%& ^
set CLASSPATH=%CLASSPATH%& ^
goto:eof
