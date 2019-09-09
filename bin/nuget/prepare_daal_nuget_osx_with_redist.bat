@echo off
rem ============================================================================
rem Copyright 2013-2019 Intel Corporation
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

set VER=2019.3.0

mkdir inteldaal.devel.osx-x64.%VER%
mkdir inteldaal.devel.osx-x64.%VER%\lib
mkdir inteldaal.devel.osx-x64.%VER%\lib\native
mkdir inteldaal.devel.osx-x64.%VER%\lib\native\include

mkdir inteldaal.static.osx-x64.%VER%
mkdir inteldaal.static.osx-x64.%VER%\lib               
mkdir inteldaal.static.osx-x64.%VER%\lib\native        
mkdir inteldaal.static.osx-x64.%VER%\lib\native\osx-x64
mkdir inteldaal.static.osx-x64.%VER%\lib\native\include

mkdir inteldaal.redist.osx-x64.%VER%
mkdir inteldaal.redist.osx-x64.%VER%\runtimes
mkdir inteldaal.redist.osx-x64.%VER%\runtimes\osx-x64
mkdir inteldaal.redist.osx-x64.%VER%\runtimes\osx-x64\native

xcopy /S /Y /Q __release_mac\daal\include inteldaal.devel.osx-x64.%VER%\lib\native\include\
xcopy /S /Y /Q __release_mac\daal\include inteldaal.static.osx-x64.%VER%\lib\native\include\

xcopy /S /Y /Q __release_mac\daal\lib\*.a inteldaal.static.osx-x64.%VER%\lib\native\osx-x64\

xcopy /S /Y /Q __release_mac\daal\lib\*.dylib  inteldaal.redist.osx-x64.%VER%\runtimes\osx-x64\native\

xcopy /Y /Q ..\documentation\en\common\license.txt inteldaal.devel.osx-x64.%VER%\
xcopy /Y /Q ..\documentation\en\common\license.txt inteldaal.redist.osx-x64.%VER%\
xcopy /Y /Q ..\documentation\en\common\license.txt inteldaal.static.osx-x64.%VER%\

xcopy /Y /Q bin\nuget\inteldaal.devel.osx-x64.nuspec  inteldaal.devel.osx-x64.%VER%\ 
xcopy /Y /Q bin\nuget\inteldaal.redist.osx-x64.nuspec inteldaal.redist.osx-x64.%VER%\
xcopy /Y /Q bin\nuget\inteldaal.static.osx-x64.nuspec inteldaal.static.osx-x64.%VER%\
