@echo off
rem ============================================================================
rem Copyright 2013-2019 Intel Corporation.
rem
rem This software  and the related  documents  are Intel  copyrighted materials,
rem and your use  of them is  governed by the express  license under  which they
rem were provided to you (License).  Unless the License provides otherwise,  you
rem may not use,  modify,  copy, publish,  distribute, disclose or transmit this
rem software or the related documents without Intel's prior written permission.
rem
rem This software and the related documents are provided as is,  with no express
rem or implied warranties,  other  than those that  are expressly  stated in the
rem License.
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
