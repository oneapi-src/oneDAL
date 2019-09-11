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

mkdir inteldaal.devel.osx-x64.%VER%
mkdir inteldaal.devel.osx-x64.%VER%\lib
mkdir inteldaal.devel.osx-x64.%VER%\lib\native
mkdir inteldaal.devel.osx-x64.%VER%\lib\native\osx-x64
mkdir inteldaal.devel.osx-x64.%VER%\lib\native\include

mkdir inteldaal.static.osx-x64.%VER%
mkdir inteldaal.static.osx-x64.%VER%\lib               
mkdir inteldaal.static.osx-x64.%VER%\lib\native        
mkdir inteldaal.static.osx-x64.%VER%\lib\native\osx-x64
mkdir inteldaal.static.osx-x64.%VER%\lib\native\include

xcopy /S /Y /Q __release_mac\daal\include inteldaal.devel.osx-x64.%VER%\lib\native\include\
xcopy /S /Y /Q __release_mac\daal\include inteldaal.static.osx-x64.%VER%\lib\native\include\

xcopy /S /Y /Q __release_mac\daal\lib\*.a inteldaal.static.osx-x64.%VER%\lib\native\osx-x64\

xcopy /S /Y /Q __release_mac\daal\lib\*.dylib  inteldaal.devel.osx-x64.%VER%\lib\native\osx-x64\

xcopy /Y /Q ..\documentation\en\common\license.txt inteldaal.devel.osx-x64.%VER%\
xcopy /Y /Q ..\documentation\en\common\license.txt inteldaal.static.osx-x64.%VER%\

xcopy /Y /Q bin\nuget\inteldaal.devel.osx-x64.nuspec  inteldaal.devel.osx-x64.%VER%\ 
xcopy /Y /Q bin\nuget\inteldaal.static.osx-x64.nuspec inteldaal.static.osx-x64.%VER%\
