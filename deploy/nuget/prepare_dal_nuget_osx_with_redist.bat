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

mkdir inteldal.devel.osx-x64.%VER%
mkdir inteldal.devel.osx-x64.%VER%\lib
mkdir inteldal.devel.osx-x64.%VER%\lib\native
mkdir inteldal.devel.osx-x64.%VER%\lib\native\include

mkdir inteldal.static.osx-x64.%VER%
mkdir inteldal.static.osx-x64.%VER%\lib               
mkdir inteldal.static.osx-x64.%VER%\lib\native        
mkdir inteldal.static.osx-x64.%VER%\lib\native\osx-x64
mkdir inteldal.static.osx-x64.%VER%\lib\native\include

mkdir inteldal.redist.osx-x64.%VER%
mkdir inteldal.redist.osx-x64.%VER%\runtimes
mkdir inteldal.redist.osx-x64.%VER%\runtimes\osx-x64
mkdir inteldal.redist.osx-x64.%VER%\runtimes\osx-x64\native

xcopy /S /Y /Q __release_mac\daal\latest\include inteldal.devel.osx-x64.%VER%\lib\native\include\
xcopy /S /Y /Q __release_mac\daal\latest\include inteldal.static.osx-x64.%VER%\lib\native\include\

xcopy /S /Y /Q __release_mac\daal\latest\lib\*.a inteldal.static.osx-x64.%VER%\lib\native\osx-x64\

xcopy /S /Y /Q __release_mac\daal\latest\lib\libonedal*.dylib  inteldal.redist.osx-x64.%VER%\runtimes\osx-x64\native\ && del /Q __release_mac\daal\latest\lib\libonedal*1.dylib

xcopy /Y /Q LICENSE inteldal.devel.osx-x64.%VER%\
xcopy /Y /Q LICENSE inteldal.redist.osx-x64.%VER%\
xcopy /Y /Q LICENSE inteldal.static.osx-x64.%VER%\

xcopy /Y /Q deploy\nuget\inteldal.devel.osx-x64.nuspec  inteldal.devel.osx-x64.%VER%\ 
xcopy /Y /Q deploy\nuget\inteldal.redist.osx-x64.nuspec inteldal.redist.osx-x64.%VER%\
xcopy /Y /Q deploy\nuget\inteldal.static.osx-x64.nuspec inteldal.static.osx-x64.%VER%\
