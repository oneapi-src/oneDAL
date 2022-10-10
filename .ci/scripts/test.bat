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

echo call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Auxiliary\Build\vcvarsall" x64
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Auxiliary\Build\vcvarsall" x64

echo call __release_win_%3\daal\latest\env\vars.bat
call __release_win_%3\daal\latest\env\vars.bat

echo set LIB=%~dp0__release_win_vc\tbb\latest\lib\intel64\vc_mt;%LIB%
set LIB=%~dp0__release_win_vc\tbb\latest\lib\intel64\vc_mt;%LIB%

echo __release_win_%3\daal\latest\examples\%1\cpp
cd __release_win_%3\daal\latest\examples\%1\cpp
echo nmake %2 compiler=%3"
nmake %2 compiler=%3
