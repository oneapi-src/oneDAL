# file: library_version_info.py
#===============================================================================
# Copyright 2014-2019 Intel Corporation.
#
# This software and the related documents are Intel copyrighted  materials,  and
# your use of  them is  governed by the  express license  under which  they were
# provided to you (License).  Unless the License provides otherwise, you may not
# use, modify, copy, publish, distribute,  disclose or transmit this software or
# the related documents without Intel's prior written permission.
#
# This software and the related documents  are provided as  is,  with no express
# or implied  warranties,  other  than those  that are  expressly stated  in the
# License.
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
