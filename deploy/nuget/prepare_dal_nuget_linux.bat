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

mkdir inteldal.devel.linux-x64.%VER%
mkdir inteldal.devel.linux-x64.%VER%\lib
mkdir inteldal.devel.linux-x64.%VER%\lib\native
mkdir inteldal.devel.linux-x64.%VER%\lib\native\linux-x64
mkdir inteldal.devel.linux-x64.%VER%\lib\native\include

mkdir inteldal.static.linux-x64.%VER%
mkdir inteldal.static.linux-x64.%VER%\lib               
mkdir inteldal.static.linux-x64.%VER%\lib\native        
mkdir inteldal.static.linux-x64.%VER%\lib\native\linux-x64
mkdir inteldal.static.linux-x64.%VER%\lib\native\include

xcopy /S /Y /Q __release_lnx\daal\latest\include inteldal.static.linux-x64.%VER%\lib\native\include\
xcopy /S /Y /Q __release_lnx\daal\latest\include inteldal.static.linux-x86.%VER%\lib\native\include\

xcopy /S /Y /Q __release_lnx\daal\latest\lib\intel64\*.a inteldal.static.linux-x64.%VER%\lib\native\linux-x64\

xcopy /S /Y /Q __release_lnx\daal\latest\lib\intel64\libonedal*.so  inteldal.devel.linux-x64.%VER%\lib\native\linux-x64\

xcopy /Y /Q LICENSE inteldal.devel.linux-x64.%VER%\
xcopy /Y /Q LICENSE inteldal.static.linux-x64.%VER%\

xcopy /Y /Q deploy\nuget\inteldal.devel.linux-x64.nuspec  inteldal.devel.linux-x64.%VER%\ 
xcopy /Y /Q deploy\nuget\inteldal.static.linux-x64.nuspec inteldal.static.linux-x64.%VER%\
