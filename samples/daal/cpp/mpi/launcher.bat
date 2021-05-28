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

if /i [%1]==[build]       (set rmode=build)       & shift
if /i [%1]==[run]         (set rmode=run)         & shift
if /i [%1]==[help]                                          goto :Usage

goto :CorrectArgs
:Usage
echo Usage: launcher.bat ^{rmode^|help^}
echo rmode - optional parameter, can be build (for building samples only) or
echo         run (for running samples only).
echo         If not specified build and run are performed.
echo help  - print this message
exit /b errorcode

:CorrectArgs

if not defined RESULT_DIR set RESULT_DIR=_results
if not exist %RESULT_DIR% md %RESULT_DIR%

echo %RESULT_DIR%

set CFLAGS=-MDd /debug:none -nologo -w
set LFLAGS=-nologo
set LIB_DAAL=onedal_cored.lib onedal_threadd.lib
set LIB_DAAL_DLL=onedal_cored_dll.lib
set LFLAGS_DAAL=%LIB_DAAL% tbb12_debug.lib tbbmalloc_debug.lib impi.lib
set LFLAGS_DAAL_DLL=onedal_cored_dll.lib
set MPI_LOGFILE=.\%RESULT_DIR%\build_mpi.log
if not "%RMODE%"=="run" (
    if exist %MPI_LOGFILE% del /Q /F %MPI_LOGFILE%
)

set MPI_CPP_PATH=sources
if not defined MPI_SAMPLE_LIST (
    call .\daal.lst.bat
)
set proc_num=4

setlocal enabledelayedexpansion enableextensions

for %%T in (%MPI_SAMPLE_LIST%) do (

    if not "%RMODE%"=="run" (
        echo call mpiicc %CFLAGS% %MPI_CPP_PATH%\%%T.cpp -Fo%RESULT_DIR%\%%T.obj %LFLAGS_DAAL% -Fe%RESULT_DIR%\%%T.exe 2>&1 >> %MPI_LOGFILE%
        call mpiicc %CFLAGS% %MPI_CPP_PATH%\%%T.cpp -Fo%RESULT_DIR%\%%T.obj %LFLAGS_DAAL% -Fe%RESULT_DIR%\%%T.exe 2>&1 >> %MPI_LOGFILE%
        echo call mpiicc %LFLAGS% %RESULT_DIR%\%%T.obj %LFLAGS_DAAL_DLL% -Fe%RESULT_DIR%\%%T_dll.exe 2>&1 >> %MPI_LOGFILE%
        call mpiicc %LFLAGS% %RESULT_DIR%\%%T.obj %LFLAGS_DAAL_DLL% -Fe%RESULT_DIR%\%%T_dll.exe 2>&1 >> %MPI_LOGFILE%
        del /F /Q %RESULT_DIR%\%%T.obj
    )

    if not "%RMODE%"=="build" (
        for %%U in (%%T %%T_dll) do (
            if exist .\%RESULT_DIR%\%%U.exe (
                echo [DEBUG] ERRORLEV: !errorlevel!
                mpiexec -localonly -n %proc_num% .\%RESULT_DIR%\%%U.exe 1>.\%RESULT_DIR%\%%U.res 2>&1
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
