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

rem %1 - Make target
rem %2 - Compiler
rem %3 - Instruction set

set exitcode=0
for /f "tokens=*" %%i in ('python -c "from multiprocessing import cpu_count; print(cpu_count())"') do set CPUCOUNT=%%i
echo CPUCOUNT=%CPUCOUNT%

echo PATH=C:\msys64\usr\bin;%PATH%
set PATH=C:\msys64\usr\bin;%PATH%

echo pacman -S --noconfirm msys/make msys/dos2unix
pacman -S --noconfirm msys/make msys/dos2unix || set exitcode=1

echo call .ci\env\tbb.bat
call .ci\env\tbb.bat || set exitcode=1

echo call .\dev\download_micromkl.bat
call .\dev\download_micromkl.bat || set exitcode=1

echo call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall" x64
call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall" x64 || set exitcode=1

echo make %1 -j%CPUCOUNT% COMPILER=%2 PLAT=win32e REQCPU=%3
make %1 -j%CPUCOUNT% COMPILER=%2 PLAT=win32e REQCPU=%3 || set exitcode=1

cmake -DINSTALL_DIR=__release_win_vc\daal\latest\lib\cmake\oneDAL -DARCH_DIR=intel64 -P cmake\scripts\generate_config.cmake || set exitcode=1
EXIT /B %exitcode%
