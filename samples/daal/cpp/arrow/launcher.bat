@echo off
rem ============================================================================
rem Copyright 2020-2021 Intel Corporation
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

::  Content:
::     Intel(R) oneAPI Data Analytics Library samples creation and run
::******************************************************************************

set ARCH=intel64
set RMODE=%1

set errorcode=0

if "%1"=="help" (
    goto :Usage
)

if not "%RMODE%"=="build" if not "%RMODE%"=="run" if not "%RMODE%"=="" (
    echo Bad second argument, must be build or run
    set errorcode=1
    goto :Usage
)

goto :CorrectArgs

:Usage
echo Usage: launcher.bat ^{help^} [rmode]
echo rmode - optional parameter, can be build (for building samples only) or
echo         run (for running samples only).
echo         If not specified build and run are performed.
echo help  - print this message
exit /b errorcode

:CorrectArgs

set RESULT_DIR=_results\%ARCH%

if not exist %RESULT_DIR% md %RESULT_DIR%

echo %RESULT_DIR%

set CFLAGS=-MDd /debug:none -nologo -w -DDAAL_CHECK_PARAMETER -std=c++14 /I %ARROWROOT%\cpp\src /I %ARROWROOT%\cpp\%ARROWCONFIG%\src /I %DAALROOT%\include
set LFLAGS=-nologo
set LIB_DAAL=onedal_cored.lib onedal_threadd.lib
set LIB_DAAL_DLL=onedal_cored_dll.lib
set LFLAGS_DAAL=%LIB_DAAL% tbb12_debug.lib tbbmalloc_debug.lib
set LFLAGS_DAAL_DLL=onedal_cored_dll.lib
set ARROW_LOGFILE=.\%RESULT_DIR%\build_arrow.log
set LIB_DOUBLE_CONV=%ARROWROOT%\cpp\%ARROWCONFIG%\double-conversion_ep\src\double-conversion_ep\lib\double-conversion.lib
set ARROW_LIBRARIES=%ARROWROOT%\cpp\%ARROWCONFIG%\%ARROWCONFIG%\Release\arrow.lib

if exist %LIB_DOUBLE_CONV% (
    set ARROW_LIBRARIES=%ARROW_LIBRARIES% %LIB_DOUBLE_CONV%
)

if not "%RMODE%"=="run" (
    if exist %ARROW_LOGFILE% del /Q /F %ARROW_LOGFILE%
)
set ARROW_CPP_PATH=sources
if not defined ARROW (
    call .\daal.lst.bat
)

setlocal enabledelayedexpansion enableextensions

for %%T in (%ARROW%) do (
    if not "%RMODE%"=="run" (
        echo call icl -c %CFLAGS% %ARROW_CPP_PATH%\%%T.cpp -Fo%RESULT_DIR%\%%T.obj 2>&1 >> %ARROW_LOGFILE%
        call      icl -c %CFLAGS% %ARROW_CPP_PATH%\%%T.cpp -Fo%RESULT_DIR%\%%T.obj 2>&1 >> %ARROW_LOGFILE%
        echo call icl %LFLAGS% %RESULT_DIR%\%%T.obj %LIB_DAAL%     %ARROW_LIBRARIES% -Fe%RESULT_DIR%\%%T.exe     2>&1 >> %ARROW_LOGFILE%
        call      icl %LFLAGS% %RESULT_DIR%\%%T.obj %LIB_DAAL%     %ARROW_LIBRARIES% -Fe%RESULT_DIR%\%%T.exe     2>&1 >> %ARROW_LOGFILE%
        echo call icl %LFLAGS% %RESULT_DIR%\%%T.obj %LIB_DAAL_DLL% %ARROW_LIBRARIES% -Fe%RESULT_DIR%\%%T_dll.exe 2>&1 >> %ARROW_LOGFILE%
        call      icl %LFLAGS% %RESULT_DIR%\%%T.obj %LIB_DAAL_DLL% %ARROW_LIBRARIES% -Fe%RESULT_DIR%\%%T_dll.exe 2>&1 >> %ARROW_LOGFILE%
    )
    if not "%RMODE%"=="build" (
        for %%U in (%%T %%T_dll) do (
            if exist .\%RESULT_DIR%\%%U.exe (
                .\%RESULT_DIR%\%%U.exe 1>.\%RESULT_DIR%\%%U.res 2>&1
                if "!errorlevel!" == "0" (
                    echo %time% PASSED %%U
                ) else (
                    echo %time% FAILED %%U with errno !errorlevel!
                )
            ) else (
                echo %time% BUILD FAILED %%U
            )
        )
    )
)

endlocal

exit /B %ERRORLEVEL%

:out
