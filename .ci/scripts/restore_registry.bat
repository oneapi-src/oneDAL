REM SPDX-FileCopyrightText: 2022 Intel Corporation
REM
REM SPDX-License-Identifier: MIT

set COMPILER_VERSION=%1
set TBB_VERSION=%2

reg add HKLM\Software\Khronos\OpenCL\Vendors /v "C:\Program Files (x86)\Intel\oneAPI\compiler\%COMPILER_VERSION%\windows\lib\x64\intelocl64.dll" /t REG_DWORD /d 0 /f
reg add HKLM\Software\Intel\oneAPI\TBB\%TBB_VERSION% /v TBB_DLL_PATH /t REG_SZ /d "C:\Program Files (x86)\Intel\oneAPI\tbb\%TBB_VERSION%\redist\intel64\vc14" /f