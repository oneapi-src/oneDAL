@echo off
rem ============================================================================
rem Copyright 2014-2019 Intel Corporation
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
::     Intel(R) Data Analytics Acceleration Library examples creation and run
::

setlocal enabledelayedexpansion enableextensions

set errorcode=0

:ParseArgs
if /i [%1]==[ia32]        (set full_ia=ia32)      & shift & goto :ParseArgs
if /i [%1]==[intel64]     (set full_ia=intel64)   & shift & goto :ParseArgs
if /i [%1]==[build]       (set rmode=build)       & shift & goto :CheckArgs
if /i [%1]==[run]         (set rmode=run)         & shift & goto :CheckArgs
if /i [%1]==[help]                                          goto :Usage

:CheckArgs
if not defined full_ia if not "%rmode%"=="build" (
   echo Bad argument ^{arch^} , must be ia32 or intel64
   set errorcode=1
   goto :Usage
)

goto :CorrectArgs

:Usage
echo Usage:  launcher.bat ^{arch^|help^} [rmode] [path_to_javac]
echo arch          - can be ia32 or intel64, optional for building examples
echo rmode         - optional parameter, can be build (for building examples only) or
echo                 run (for running examples only).
echo                 If not specified build and run are performed.
echo path_to_javac - optional parameter.
echo                 Specify it in case, if you do not want to use default javac
echo help          - print this message
echo Example: launcher.bat ia32 build "C:\Program Files\Java\jdk1.8.0_20" or launcher.bat intel64 run
echo Be sure, that you put a path to javac into semicolons
exit /b errorcode

:CorrectArgs

set class_path=com\intel\daal\examples

:: Setting environment for side javac if the path specified
set path_to_javac=%1%

if defined path_to_javac (
    set PATH=%path_to_javac%\bin;"%PATH%"
    set LIB=%path_to_javac%\lib;"%LIB%"
    set INCLUDE=%path_to_javac%\include;"%INCLUDE%"
)

:: Setting list of Java examples to process
if not defined Java_example_list (
    call .\daal.lst.bat
)

:: Setting path for JavaAPI library
set Djava_library_path="%DAALROOT%"\..\redist\%full_ia%\daal
set Djava_library_path="%DAALROOT%"\redist\%full_ia%;%Djava_library_path%

:: Setting a path for result folder to put results of examples in
if defined full_ia (
    set result_folder=_results\%full_ia%
    if exist !result_folder! rd /S /Q !result_folder!
    md !result_folder!
)

for %%A in (%Java_example_list%) do (
:: Building java examples from the list and writing info in the build log
    if not "%rmode%"=="run" (
        javac %class_path%\%%A.java
    )
:: Running examples and putting the result in example's result file in res folder
    if not "%rmode%"=="build" (
        if not exist "%class_path%\%%A.class" (
            echo !time! BUILD FAILED %%A
        ) else (
            for /f "tokens=1,2,3 delims=\" %%a in ("%%A") do (
                if "%%c"=="" (
                    if not exist !result_folder!\%%a md !result_folder!\%%a

                    if "%full_ia%"=="intel64" ( set memory=4g ) else ( set memory=1g )
                    java -Xmx!memory! -Djava.library.path=%Djava_library_path% com.intel.daal.examples.%%a.%%b > !result_folder!\%%a\%%b.res

                    set errorcode=!ERRORLEVEL!
                    if !errorcode! == 0 ( echo !time! PASSED %%b ) else ( echo !time! FAILED %%b with errno !errorcode!)
                ) else (
                    if not exist !result_folder!\%%a md !result_folder!\%%a
                    if not exist !result_folder!\%%a\%%b md !result_folder!\%%a\%%b

                    if "%full_ia%"=="intel64" ( set memory=2g ) else ( set memory=1g )
                    java -Xmx!memory! -Djava.library.path=%Djava_library_path% com.intel.daal.examples.%%a.%%b.%%c > !result_folder!\%%a\%%b\%%c.res

                    set errorcode=!ERRORLEVEL!
                    if !errorcode! == 0 ( echo !time! PASSED %%c ) else ( echo !time! FAILED %%c with errno !errorcode! )
                )
            )
        )
    )
)
endlocal
