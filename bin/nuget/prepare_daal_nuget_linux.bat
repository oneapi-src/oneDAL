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
