@echo off
rem ============================================================================
rem Copyright 2017-2019 Intel Corporation.
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
rem
rem License:
rem http://software.intel.com/en-us/articles/intel-sample-source-code-license-a
rem greement/
rem ============================================================================

::  Content:
::     Intel(R) Data Analytics Acceleration Library samples creation and run
::******************************************************************************

set ARCH=intel64
set RMODE=%1

set errorcode=0

if "%1"=="help" (
    goto :Usage
)

if not "%RMODE%"=="build" if not "%RMODE%"=="run" if not "%RMODE%"=="" (
    echo Bad first argument, must be build or run
    set errorcode=1
    goto :Usage
)

goto :CorrectArgs

:Usage
echo Usage: launcher.bat ^{rmode^|help^}
echo rmode - optional parameter, can be build (for building samples only) or
echo         run (for running samples only).
echo         If not specified build and run are performed.
echo help  - print this message
exit /b errorcode

:CorrectArgs

if not defined RESULT_DIR set RESULT_DIR=_results\%ARCH%
if not exist %RESULT_DIR% md %RESULT_DIR%

echo %RESULT_DIR%

set CFLAGS=-nologo -w
set LFLAGS=-nologo
set LIB_DAAL=daal_core.lib daal_thread.lib
set LIB_DAAL_DLL=daal_core_dll.lib
set LFLAGS_DAAL=%LIB_DAAL% tbb.lib tbbmalloc.lib impi.lib
set LFLAGS_DAAL_DLL=daal_core_dll.lib
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
        echo call mpiicc -c %CFLAGS% %MPI_CPP_PATH%\%%T.cpp %LIB_DAAL% -Fo%RESULT_DIR%\%%T.obj 2>&1 >> %MPI_LOGFILE%
        call mpiicc -c %CFLAGS% %MPI_CPP_PATH%\%%T.cpp %LIB_DAAL% -Fo%RESULT_DIR%\%%T.obj 2>&1 >> %MPI_LOGFILE%
        echo call mpiicc %LFLAGS% %RESULT_DIR%\%%T.obj %LFLAGS_DAAL% -Fe%RESULT_DIR%\%%T.exe 2>&1 >> %MPI_LOGFILE%
        call mpiicc %LFLAGS% %RESULT_DIR%\%%T.obj %LFLAGS_DAAL% -Fe%RESULT_DIR%\%%T.exe 2>&1 >> %MPI_LOGFILE%
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
