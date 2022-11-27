@echo off
rem ============================================================================
rem Copyright 2022 Intel Corporation
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

rem %1 - Examples target
rem %2 - Examples linking
rem %3 - Compiler

for /f "tokens=*" %%i in ('python -c "from multiprocessing import cpu_count; print(cpu_count())"') do set CPUCOUNT=%%i
echo CPUCOUNT=%CPUCOUNT%

echo PATH=C:\msys64\usr\bin;%PATH%
set PATH=C:\msys64\usr\bin;%PATH%

echo call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall" x64
call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall" x64

echo call __release_win_vc\daal\latest\env\vars.bat
call __release_win_vc\daal\latest\env\vars.bat

echo set LIB=%~dp0..\..\__release_win_vc\tbb\latest\lib\intel64\vc_mt;%LIB%
set LIB=%~dp0..\..\__release_win_vc\tbb\latest\lib\intel64\vc_mt;%LIB%
echo set PATH=%~dp0..\..\__release_win_vc\tbb\latest\lib\intel64\vc_mt;%PATH%
set PATH=%~dp0..\..\__release_win_vc\tbb\latest\lib\intel64\vc_mt;%PATH%

echo set LIB=%~dp0..\..\__release_win_vc\tbb\latest\redist\intel64\vc_mt;%LIB%
set LIB=%~dp0..\..\__release_win_vc\tbb\latest\redist\intel64\vc_mt;%LIB%
echo set PATH=%~dp0..\..\__release_win_vc\tbb\latest\redist\intel64\vc_mt;%PATH%
set PATH=%~dp0..\..\__release_win_vc\tbb\latest\redist\intel64\vc_mt;%PATH%

echo __release_win_vc\daal\latest\examples\%1
cd __release_win_vc\daal\latest\examples\%1
if not "%1"=="daal\java" (
    echo nmake %2 compiler=%3
    nmake %2 compiler=%3
) else (
    echo Java installation
    echo JAVA_HOME=%JAVA_HOME_17_X64%
    set JAVA_HOME=%JAVA_HOME_17_X64%
    echo PATH=%JAVA_HOME%\bin;%PATH%
    set PATH=%JAVA_HOME%\bin;%PATH%
    echo set INCLUDE=%JAVA_HOME%\include;%JAVA_HOME%\include\win32;%INCLUDE%
    set INCLUDE=%JAVA_HOME%\include;%JAVA_HOME%\include\win32;%INCLUDE%
    call launcher.bat
)
