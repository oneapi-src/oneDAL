/* file: error_handling_throw.cpp */
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
 * <a name="DAAL-EXAMPLE-CPP-ERROR_HANDLING_THROW"></a>
 * \example error_handling_throw.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;

std::string wrongDatasetFileName = "../data/batch/wrong.csv";

int main(int argc, char *argv[])
{
    try
    {
        /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
        FileDataSource<CSVFeatureManager> wrongDataSource(wrongDatasetFileName);
        /* An exception was generated due to absense DAAL_NOTHROW_EXCEPTIONS define by default */
    }
    catch(daal::services::Exception e)
    {
        /* Retrieve the description of the generated exception. */
        std::cout << "FileDataSource expected error: " << e.what() << std::endl;
    }

    return 0;
}
