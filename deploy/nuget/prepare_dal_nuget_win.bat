@echo off
rem ============================================================================
rem Copyright 2013-2021 Intel Corporation
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

set VER=2021.5.0

mkdir inteldal.devel.win-x64.%VER%
mkdir inteldal.devel.win-x64.%VER%\lib
mkdir inteldal.devel.win-x64.%VER%\lib\native
mkdir inteldal.devel.win-x64.%VER%\lib\native\win-x64
mkdir inteldal.devel.win-x64.%VER%\lib\native\include
mkdir inteldal.devel.win-x64.%VER%\build
mkdir inteldal.devel.win-x64.%VER%\build\native  

mkdir inteldal.static.win-x64.%VER%
mkdir inteldal.static.win-x64.%VER%\lib               
mkdir inteldal.static.win-x64.%VER%\lib\native        
mkdir inteldal.static.win-x64.%VER%\lib\native\win-x64
mkdir inteldal.static.win-x64.%VER%\lib\native\include
mkdir inteldal.static.win-x64.%VER%\build             
mkdir inteldal.static.win-x64.%VER%\build\native         

mkdir inteldal.redist.win-x64.%VER%
mkdir inteldal.redist.win-x64.%VER%\runtimes
mkdir inteldal.redist.win-x64.%VER%\runtimes\win-x64
mkdir inteldal.redist.win-x64.%VER%\runtimes\win-x64\native
mkdir inteldal.redist.win-x64.%VER%\build       
mkdir inteldal.redist.win-x64.%VER%\build\native         

xcopy /S /Y /Q __release_win\daal\latest\include inteldal.devel.win-x64.%VER%\lib\native\include\
xcopy /S /Y /Q __release_win\daal\latest\include inteldal.static.win-x64.%VER%\lib\native\include\

xcopy /S /Y /Q __release_win\daal\latest\lib\intel64\*_dll* inteldal.devel.win-x64.%VER%\lib\native\win-x64\
xcopy /S /Y /Q __release_win\daal\latest\lib\intel64 inteldal.static.win-x64.%VER%\lib\native\win-x64\ && del /Q inteldal.static.win-x64.%VER%\lib\native\win-x64\*_dll*

xcopy /S /Y /Q __release_win\daal\latest\redist\intel64\onedal*.1.dll inteldal.redist.win-x64.%VER%\runtimes\win-x64\native\

xcopy /Y /Q LICENSE inteldal.devel.win-x64.%VER%\
xcopy /Y /Q LICENSE inteldal.redist.win-x64.%VER%\
xcopy /Y /Q LICENSE inteldal.static.win-x64.%VER%\

xcopy /Y /Q deploy\nuget\inteldal.devel.win-x64.nuspec  inteldal.devel.win-x64.%VER%\ 
xcopy /Y /Q deploy\nuget\inteldal.redist.win-x64.nuspec inteldal.redist.win-x64.%VER%\
xcopy /Y /Q deploy\nuget\inteldal.static.win-x64.nuspec inteldal.static.win-x64.%VER%\

xcopy /Y /Q deploy\nuget\inteldal.static.win-x64.targets inteldal.static.win-x64.%VER%\build\native\
xcopy /Y /Q deploy\nuget\inteldal.static.win-x64.xml     inteldal.static.win-x64.%VER%\build\native\

xcopy /Y /Q deploy\nuget\inteldal.devel.win-x64.targets inteldal.devel.win-x64.%VER%\build\native\
xcopy /Y /Q deploy\nuget\inteldal.devel.win-x64.xml     inteldal.devel.win-x64.%VER%\build\native\

xcopy /Y /Q deploy\nuget\inteldal.redist.win-x64.targets inteldal.redist.win-x64.%VER%\build\native\
