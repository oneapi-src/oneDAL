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

mkdir inteldaal.devel.win-x64.%VER%
mkdir inteldaal.devel.win-x64.%VER%\lib
mkdir inteldaal.devel.win-x64.%VER%\lib\native
mkdir inteldaal.devel.win-x64.%VER%\lib\native\win-x64
mkdir inteldaal.devel.win-x64.%VER%\lib\native\include
mkdir inteldaal.devel.win-x64.%VER%\build
mkdir inteldaal.devel.win-x64.%VER%\build\native
mkdir inteldaal.devel.win-x86.%VER%
mkdir inteldaal.devel.win-x86.%VER%\lib               
mkdir inteldaal.devel.win-x86.%VER%\lib\native        
mkdir inteldaal.devel.win-x86.%VER%\lib\native\win-x86
mkdir inteldaal.devel.win-x86.%VER%\lib\native\include
mkdir inteldaal.devel.win-x86.%VER%\build             
mkdir inteldaal.devel.win-x86.%VER%\build\native      

mkdir inteldaal.static.win-x64.%VER%
mkdir inteldaal.static.win-x64.%VER%\lib               
mkdir inteldaal.static.win-x64.%VER%\lib\native        
mkdir inteldaal.static.win-x64.%VER%\lib\native\win-x64
mkdir inteldaal.static.win-x64.%VER%\lib\native\include
mkdir inteldaal.static.win-x64.%VER%\build             
mkdir inteldaal.static.win-x64.%VER%\build\native      
mkdir inteldaal.static.win-x86.%VER%
mkdir inteldaal.static.win-x86.%VER%\lib               
mkdir inteldaal.static.win-x86.%VER%\lib\native        
mkdir inteldaal.static.win-x86.%VER%\lib\native\win-x86
mkdir inteldaal.static.win-x86.%VER%\lib\native\include
mkdir inteldaal.static.win-x86.%VER%\build             
mkdir inteldaal.static.win-x86.%VER%\build\native      

mkdir inteldaal.redist.win-x64.%VER%
mkdir inteldaal.redist.win-x64.%VER%\runtimes
mkdir inteldaal.redist.win-x64.%VER%\runtimes\win-x64
mkdir inteldaal.redist.win-x64.%VER%\runtimes\win-x64\native
mkdir inteldaal.redist.win-x64.%VER%\build       
mkdir inteldaal.redist.win-x64.%VER%\build\native
mkdir inteldaal.redist.win-x86.%VER%
mkdir inteldaal.redist.win-x86.%VER%\runtimes               
mkdir inteldaal.redist.win-x86.%VER%\runtimes\win-x86       
mkdir inteldaal.redist.win-x86.%VER%\runtimes\win-x86\native
mkdir inteldaal.redist.win-x86.%VER%\build                  
mkdir inteldaal.redist.win-x86.%VER%\build\native           

xcopy /S /Y /Q __release_win\daal\include inteldaal.devel.win-x64.%VER%\lib\native\include\
xcopy /S /Y /Q __release_win\daal\include inteldaal.devel.win-x86.%VER%\lib\native\include\
xcopy /S /Y /Q __release_win\daal\include inteldaal.static.win-x64.%VER%\lib\native\include\
xcopy /S /Y /Q __release_win\daal\include inteldaal.static.win-x86.%VER%\lib\native\include\

xcopy /S /Y /Q __release_win\daal\lib\intel64_win\*_dll* inteldaal.devel.win-x64.%VER%\lib\native\win-x64\
xcopy /S /Y /Q __release_win\daal\lib\intel64_win inteldaal.static.win-x64.%VER%\lib\native\win-x64\  && del /Q inteldaal.static.win-x64.%VER%\lib\native\win-x64\*_dll*
xcopy /S /Y /Q __release_win\daal\lib\ia32_win\*_dll* inteldaal.devel.win-x86.%VER%\lib\native\win-x86\
xcopy /S /Y /Q __release_win\daal\lib\ia32_win inteldaal.static.win-x86.%VER%\lib\native\win-x86\     && del /Q inteldaal.static.win-x86.%VER%\lib\native\win-x86\*_dll*

xcopy /S /Y /Q __release_win\redist\intel64_win\daal inteldaal.redist.win-x64.%VER%\runtimes\win-x64\native\
xcopy /S /Y /Q __release_win\redist\ia32_win\daal inteldaal.redist.win-x86.%VER%\runtimes\win-x86\native\

xcopy /Y /Q ..\documentation\en\common\license.txt inteldaal.devel.win-x64.%VER%\
xcopy /Y /Q ..\documentation\en\common\license.txt inteldaal.devel.win-x86.%VER%\
xcopy /Y /Q ..\documentation\en\common\license.txt inteldaal.redist.win-x64.%VER%\
xcopy /Y /Q ..\documentation\en\common\license.txt inteldaal.redist.win-x86.%VER%\
xcopy /Y /Q ..\documentation\en\common\license.txt inteldaal.static.win-x64.%VER%\
xcopy /Y /Q ..\documentation\en\common\license.txt inteldaal.static.win-x86.%VER%\

xcopy /Y /Q bin\nuget\inteldaal.devel.win-x64.nuspec  inteldaal.devel.win-x64.%VER%\ 
xcopy /Y /Q bin\nuget\inteldaal.devel.win-x86.nuspec  inteldaal.devel.win-x86.%VER%\ 
xcopy /Y /Q bin\nuget\inteldaal.redist.win-x64.nuspec inteldaal.redist.win-x64.%VER%\
xcopy /Y /Q bin\nuget\inteldaal.redist.win-x86.nuspec inteldaal.redist.win-x86.%VER%\
xcopy /Y /Q bin\nuget\inteldaal.static.win-x64.nuspec inteldaal.static.win-x64.%VER%\
xcopy /Y /Q bin\nuget\inteldaal.static.win-x86.nuspec inteldaal.static.win-x86.%VER%\

xcopy /Y /Q bin\nuget\inteldaal.static.win-x64.targets inteldaal.static.win-x64.%VER%\build\native\
xcopy /Y /Q bin\nuget\inteldaal.static.win-x64.xml     inteldaal.static.win-x64.%VER%\build\native\
xcopy /Y /Q bin\nuget\inteldaal.static.win-x86.targets inteldaal.static.win-x86.%VER%\build\native\
xcopy /Y /Q bin\nuget\inteldaal.static.win-x86.xml     inteldaal.static.win-x86.%VER%\build\native\

xcopy /Y /Q bin\nuget\inteldaal.devel.win-x64.targets inteldaal.devel.win-x64.%VER%\build\native\
xcopy /Y /Q bin\nuget\inteldaal.devel.win-x64.xml     inteldaal.devel.win-x64.%VER%\build\native\
xcopy /Y /Q bin\nuget\inteldaal.devel.win-x86.targets inteldaal.devel.win-x86.%VER%\build\native\
xcopy /Y /Q bin\nuget\inteldaal.devel.win-x86.xml     inteldaal.devel.win-x86.%VER%\build\native\

xcopy /Y /Q bin\nuget\inteldaal.redist.win-x64.targets inteldaal.redist.win-x64.%VER%\build\native\
xcopy /Y /Q bin\nuget\inteldaal.redist.win-x86.targets inteldaal.redist.win-x86.%VER%\build\native\
