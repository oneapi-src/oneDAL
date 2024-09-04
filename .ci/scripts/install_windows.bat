@echo off
rem ============================================================================
rem Copyright 2024 Intel Corporation
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

set URL=%1
set COMPONENTS=%2

curl.exe --output %TEMP%\webimage.exe --url %URL% --retry 5 --retry-delay 5
start /b /wait %TEMP%\webimage.exe -s -x -f webimage_extracted --log extract.log
del %TEMP%\webimage.exe
if "%COMPONENTS%"=="" (
  webimage_extracted\bootstrapper.exe -s --action install --eula=accept --install-dir=C:\temp\oneapi\ -p=NEED_VS2017_INTEGRATION=1 -p=NEED_VS2019_INTEGRATION=1 -p=NEED_VS2022_INTEGRATION=1 --log-dir=.
) else (
  webimage_extracted\bootstrapper.exe -s --action install --components=%COMPONENTS% --eula=accept --install-dir=C:\temp\oneapi\ -p=NEED_VS2017_INTEGRATION=1 -p=NEED_VS2019_INTEGRATION=1 -p=NEED_VS2022_INTEGRATION=1 --log-dir=.
)
set installer_exit_code=%ERRORLEVEL%
rd /s/q "webimage_extracted"
exit /b %installer_exit_code%
