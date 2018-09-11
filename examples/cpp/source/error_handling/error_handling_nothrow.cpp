/* file: error_handling_nothrow.cpp */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
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
!    C++ example of error handling mechanism without throwing exceptions
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-ERROR_HANDLING_NOTHROW"></a>
 * \example error_handling_nothrow.cpp
 */

#define DAAL_NOTHROW_EXCEPTIONS

#include "daal.h"
#include "service.h"

using namespace daal;

std::string wrongDatasetFileName = "../data/batch/wrong.csv";

int main(int argc, char *argv[])
{
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> wrongDataSource(wrongDatasetFileName);
    /* No exception was generated due to DAAL_NOTHROW_EXCEPTIONS define */

    /* Retrieve errors from FileDataSource<CSVFeatureManager> and their description. */
    std::cout << "FileDataSource expected error: " << wrongDataSource.status().getDescription() << std::endl;

    return 0;
}
