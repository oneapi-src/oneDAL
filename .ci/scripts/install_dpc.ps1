#===============================================================================
# Copyright contributors to the oneDAL project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================
echo "Download intel DPC++ compiler"
(new-object System.Net.WebClient).DownloadFile("https://registrationcenter-download.intel.com/akdlm/IRC_NAS/7991e201-ca0f-4689-bdb6-1ed73a8246fd/w_dpcpp-cpp-compiler_p_2024.2.0.491_offline.exe", "dpcpp_installer.exe")
# wget -q -O dpcpp_installer.exe 
echo "Unpacking DPC++ installer"
Start-Process ".\dpcpp_installer.exe" -ArgumentList "--s --x --f oneAPI" -Wait
echo "Installing DPC++ compiler"
# Installing the compiler can take upwards of 20 minutes
# It does not print any messages during installation
Start-Process ".\oneAPI\bootstrapper.exe" -ArgumentList "-s --eula=accept --install-dir=dpcpp" -Wait
echo "remove installer files"
Remove-Item -LiteralPath .\oneAPI -Force -Recurse -ErrorAction Ignore
Remove-Item .\dpcpp_installer.exe -Force
echo "DPC++ install complete"
