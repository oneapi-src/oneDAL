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

mkdir inteldaal.devel.linux-x64.%VER%
mkdir inteldaal.devel.linux-x64.%VER%\lib
mkdir inteldaal.devel.linux-x64.%VER%\lib\native
mkdir inteldaal.devel.linux-x64.%VER%\lib\native\linux-x64
mkdir inteldaal.devel.linux-x64.%VER%\lib\native\include
mkdir inteldaal.devel.linux-x86.%VER%
mkdir inteldaal.devel.linux-x86.%VER%\lib               
mkdir inteldaal.devel.linux-x86.%VER%\lib\native        
mkdir inteldaal.devel.linux-x86.%VER%\lib\native\linux-x86
mkdir inteldaal.devel.linux-x86.%VER%\lib\native\include

mkdir inteldaal.static.linux-x64.%VER%
mkdir inteldaal.static.linux-x64.%VER%\lib               
mkdir inteldaal.static.linux-x64.%VER%\lib\native        
mkdir inteldaal.static.linux-x64.%VER%\lib\native\linux-x64
mkdir inteldaal.static.linux-x64.%VER%\lib\native\include
mkdir inteldaal.static.linux-x86.%VER%
mkdir inteldaal.static.linux-x86.%VER%\lib               
mkdir inteldaal.static.linux-x86.%VER%\lib\native        
mkdir inteldaal.static.linux-x86.%VER%\lib\native\linux-x86
mkdir inteldaal.static.linux-x86.%VER%\lib\native\include

xcopy /S /Y /Q __release_lnx\daal\include inteldaal.devel.linux-x64.%VER%\lib\native\include\
xcopy /S /Y /Q __release_lnx\daal\include inteldaal.devel.linux-x86.%VER%\lib\native\include\
xcopy /S /Y /Q __release_lnx\daal\include inteldaal.static.linux-x64.%VER%\lib\native\include\
xcopy /S /Y /Q __release_lnx\daal\include inteldaal.static.linux-x86.%VER%\lib\native\include\

xcopy /S /Y /Q __release_lnx\daal\lib\intel64_lin\*.a inteldaal.static.linux-x64.%VER%\lib\native\linux-x64\
xcopy /S /Y /Q __release_lnx\daal\lib\ia32_lin\*.a    inteldaal.static.linux-x86.%VER%\lib\native\linux-x86\

xcopy /S /Y /Q __release_lnx\daal\lib\intel64_lin\*.so  inteldaal.devel.linux-x64.%VER%\lib\native\linux-x64\
xcopy /S /Y /Q __release_lnx\daal\lib\ia32_lin\*.so     inteldaal.devel.linux-x86.%VER%\lib\native\linux-x86\

xcopy /Y /Q ..\documentation\en\common\license.txt inteldaal.devel.linux-x64.%VER%\
xcopy /Y /Q ..\documentation\en\common\license.txt inteldaal.devel.linux-x86.%VER%\
xcopy /Y /Q ..\documentation\en\common\license.txt inteldaal.static.linux-x64.%VER%\
xcopy /Y /Q ..\documentation\en\common\license.txt inteldaal.static.linux-x86.%VER%\

xcopy /Y /Q bin\nuget\inteldaal.devel.linux-x64.nuspec  inteldaal.devel.linux-x64.%VER%\ 
xcopy /Y /Q bin\nuget\inteldaal.devel.linux-x86.nuspec  inteldaal.devel.linux-x86.%VER%\ 
xcopy /Y /Q bin\nuget\inteldaal.static.linux-x64.nuspec inteldaal.static.linux-x64.%VER%\
xcopy /Y /Q bin\nuget\inteldaal.static.linux-x86.nuspec inteldaal.static.linux-x86.%VER%\
