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

echo "Download intel opencl runtime"
wget -q -O opencl_installer.exe https://registrationcenter-download.intel.com/akdlm/IRC_NAS/d9883ab0-0e26-47fd-9612-950b95460d72/w_opencl_runtime_p_2024.2.0.980.exe
echo "Unpacking opencl runtime installer"
Start-Process ".\opencl_installer.exe" -ArgumentList "--s --x --f ocl" -Wait
Move-Item -Path ".\ocl\w_opencl_runtime_p_2024.2.0.980.msi" -Destination ".\opencl_rt.msi"