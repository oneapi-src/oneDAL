@echo off
rem ============================================================================
rem Copyright contributors to the oneDAL project
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

rem %1 - url to download
rem %2 - components to download (following oneapi standards, default: default)
rem %3 - install directory (default: C:\temp\oneapi\)

set URL=%1
if "%2"=="" (set COMPONENTS=default) else (set COMPONENTS=%2)
if "%3"=="" (set DIRECTORY=C:\temp\oneapi\) else (set DIRECTORY=%3)

echo test
echo %DIRECTORY%
echo test

curl.exe --output %TEMP%\webimage.exe --url %URL% --retry 5 --retry-delay 5
start /b /wait %TEMP%\webimage.exe -s -x -f webimage_extracted --log extract.log
del %TEMP%\webimage.exe
webimage_extracted\bootstrapper.exe -s --action install --components %COMPONENTS% --eula accept --install-dir %DIRECTORY% -p=NEED_VS2017_INTEGRATION=0 -p=NEED_VS2019_INTEGRATION=0 -p=NEED_VS2022_INTEGRATION=0 --log-dir=.
set installer_exit_code=%ERRORLEVEL%
rd /s/q "webimage_extracted"
exit /b %installer_exit_code%
