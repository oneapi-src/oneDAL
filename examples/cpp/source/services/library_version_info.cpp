/* file: library_version_info.cpp */
/*******************************************************************************
* Copyright 2015-2019 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/*
!  Content:
!    Intel(R) DAAL version information
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-LIBRARY_VERSION_INFO"></a>
 * \example library_version_info.cpp
 */

#include "daal.h"
#include <iostream>

using namespace std;
using namespace daal::services;

int main(int argc, char *argv[])
{
    LibraryVersionInfo ver;

    std::cout << "Major version:          " << ver.majorVersion  << std::endl;
    std::cout << "Minor version:          " << ver.minorVersion  << std::endl;
    std::cout << "Update version:         " << ver.updateVersion << std::endl;
    std::cout << "Product status:         " << ver.productStatus << std::endl;
    std::cout << "Build:                  " << ver.build         << std::endl;
    std::cout << "Build revision:         " << ver.build_rev     << std::endl;
    std::cout << "Name:                   " << ver.name          << std::endl;
    std::cout << "Processor optimization: " << ver.processor     << std::endl;

    return 0;
}
