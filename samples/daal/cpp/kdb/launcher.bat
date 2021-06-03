@echo off
rem ============================================================================
rem Copyright 2017-2021 Intel Corporation
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

setlocal enabledelayedexpansion enableextensions

set errorcode=0

:ParseArgs
if /i [%1]==[ia32]        (echo 32-bit version is not supporterd)      & shift & goto :Usage
if /i [%1]==[intel64]     (set full_ia=intel64)   & shift & goto :ParseArgs
if /i [%1]==[build]       (set rmode=build)       & shift
if /i [%1]==[run]         (set rmode=run)         & shift
if /i [%1]==[help]                                          goto :Usage

goto :CorrectArgs

:Usage
echo Usage: launcher.bat [help] [rmode]
echo rmode - optional parameter, can be build (for building samples only) or
echo         run (for running samples only).
echo         If not specified build and run are performed.
echo help  - print this message
exit /b errorcode

:CorrectArgs

set RESULT_DIR=_results

if not exist %RESULT_DIR% md %RESULT_DIR%

echo %RESULT_DIR%

set CFLAGS=-MD -nologo -w -DDAAL_CHECK_PARAMETER /I %KDB_HEADER_PATH%
set LFLAGS=-nologo
set LIB_DAAL=onedal_core.lib onedal_thread.lib
set LIB_DAAL_DLL=onedal_core_dll.lib
set LFLAGS_DAAL=%LIB_DAAL% tbb12.lib tbbmalloc.lib impi.lib
set LFLAGS_DAAL_DLL=onedal_core_dll.lib
set KDB_LOGFILE=.\%RESULT_DIR%\build_kdb.log
if not "%RMODE%"=="run" (
    if exist %KDB_LOGFILE% del /Q /F %KDB_LOGFILE%
)
set KDB_CPP_PATH=sources
if not defined KDB_SAMPLE_LIST (
    call .\daal.lst.bat
)

setlocal enabledelayedexpansion enableextensions

for %%T in (%KDB_SAMPLE_LIST%) do (
    if not "%RMODE%"=="run" (
        echo call icl -c %CFLAGS% %KDB_CPP_PATH%\%%T.cpp -Fo%RESULT_DIR%\%%T.obj 2>&1 >> %KDB_LOGFILE%
        call      icl -c %CFLAGS% %KDB_CPP_PATH%\%%T.cpp -Fo%RESULT_DIR%\%%T.obj 2>&1 >> %KDB_LOGFILE%
        echo call icl %LFLAGS% %RESULT_DIR%\%%T.obj %LIB_DAAL%     %KDB_LIBRARY_PATH%\c.lib ws2_32.lib -Fe%RESULT_DIR%\%%T.exe     2>&1 >> %KDB_LOGFILE%
        call      icl %LFLAGS% %RESULT_DIR%\%%T.obj %LIB_DAAL%     %KDB_LIBRARY_PATH%\c.lib ws2_32.lib -Fe%RESULT_DIR%\%%T.exe     2>&1 >> %KDB_LOGFILE%
        echo call icl %LFLAGS% %RESULT_DIR%\%%T.obj %LIB_DAAL_DLL% %KDB_LIBRARY_PATH%\c.lib ws2_32.lib -Fe%RESULT_DIR%\%%T_dll.exe 2>&1 >> %KDB_LOGFILE%
        call      icl %LFLAGS% %RESULT_DIR%\%%T.obj %LIB_DAAL_DLL% %KDB_LIBRARY_PATH%\c.lib ws2_32.lib -Fe%RESULT_DIR%\%%T_dll.exe 2>&1 >> %KDB_LOGFILE%
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
