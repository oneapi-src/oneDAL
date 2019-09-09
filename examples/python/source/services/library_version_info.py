# file: library_version_info.py
#===============================================================================
# Copyright 2014-2019 Intel Corporation
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

## <a name="DAAL-EXAMPLE-PY-LIBRARY_VERSION_INFO"></a>
## \example library_version_info.py

from daal.services import LibraryVersionInfo

if __name__ == "__main__":

    ver = LibraryVersionInfo()

    print("Major version:          {}".format(ver.majorVersion))
    print("Minor version:          {}".format(ver.minorVersion))
    print("Update version:         {}".format(ver.updateVersion))
    print("Product status:         {}".format(ver.productStatus))
    print("Build:                  {}".format(ver.build))
    print("Build revision:         {}".format(ver.build_rev))
    print("Name:                   {}".format(ver.name))
    print("Processor optimization: {}".format(ver.processor))
