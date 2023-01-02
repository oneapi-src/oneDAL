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

setlocal enabledelayedexpansion enableextensions

rem %1 - Examples target
rem %2 - Examples linking
rem %3 - Compiler
rem %4 - build system
set examples=%1
set linking=%2
set compiler=%3
set build_system=%4

set errorcode=0

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

echo set TBB_DIR=%~dp0..\..\__deps\tbb\win\tbb\lib\cmake\tbb
set TBB_DIR=%~dp0..\..\__deps\tbb\win\tbb\lib\cmake\tbb

echo Java installation
echo JAVA_HOME=%JAVA_HOME_17_X64%
set JAVA_HOME=%JAVA_HOME_17_X64%
echo PATH=%JAVA_HOME%\bin;%PATH%
set PATH=%JAVA_HOME%\bin;%PATH%
echo set INCLUDE=%JAVA_HOME%\include;%JAVA_HOME%\include\win32;%INCLUDE%
set INCLUDE=%JAVA_HOME%\include;%JAVA_HOME%\include\win32;%INCLUDE%

echo __release_win_vc\daal\latest\examples\%examples%
cd __release_win_vc\daal\latest\examples\%examples%

set cmake_link_mode_short=so
set cmake_link_mode=dynamic
if "%link_mode%"=="lib" (
    set cmake_link_mode_short=a
    set cmake_link_mode=static
)

if "%examples%"=="daal\java" call launcher.bat intel64
if "%build_system%"=="cmake" (
    if exist Build rd /S /Q Build
    md Build
    cd Build
    set results_dir=_cmake_results\intel_intel64_%cmake_link_mode_short%\Release
    echo cmake -DTARGET_LINK=%cmake_link_mode% -DTBB_DIR=%TBB_DIR% ..
    cmake -DTARGET_LINK=%cmake_link_mode% -DTBB_DIR=%TBB_DIR% ..
    set solution_name=%examples:\=_%
    msbuild.exe "!solution_name!_examples.sln" /p:Configuration=Release

    echo "List of examples built in !results_dir!:"
    dir /B !results_dir!\

    for /f "delims=." %%F in ('dir /B !results_dir!\*.exe 2^> nul') do (
        set example=%%F
        echo !example! >> cmake_examples_list.txt
    )

    for /f %%G in (cmake_examples_list.txt) do (
        set ExampleName=%%G
        set exe_log=%%G.res
        if exist !results_dir!\%%G.exe (
            !results_dir!\%%G.exe
            set errorcode=!errorlevel!
            if !errorcode! EQU 0 (
                set status_ex=PASSED %ExampleName%
                echo !status_ex!
            ) else (
                echo FAILED %ExampleName% with errno !errorcode!
            )
        )
    )
) else (
    if "%examples%"=="daal\cpp" nmake %linking% compiler=%compiler%
    if "%examples%"=="oneapi\cpp" nmake %linking% compiler=%compiler%
)
