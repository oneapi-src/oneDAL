/* file: error_handling.h */
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
!    Auxiliary error-handling functions used in C++ examples
!******************************************************************************/


#ifndef _ERROR_HANDLING_H
#define _ERROR_HANDLING_H

#include <daal.h>

const int fileError = -1001;

void checkAllocation(void *ptr)
{
    if (!ptr)
    {
        std::cout << "Error: Memory allocation failed" << std::endl;
        exit(-1);
    }
}

void checkPtr(void *ptr)
{
    if (!ptr)
    {
        std::cout << "Error: NULL pointer" << std::endl;
        exit(-2);
    }
}

void fileOpenError(const char *filename)
{
    std::cout << "Unable to open file '" << filename << "'" << std::endl;
    exit(fileError);
}

void fileReadError()
{
    std::cout << "Unable to read next line" << std::endl;
    exit(fileError);
}

void sparceFileReadError()
{
    std::cout << "Incorrect format of file" << std::endl;
    exit(fileError);
}

void checkStatus(const daal::services::Status &s)
{
    if (!s)
    {
        std::cout << s.getDescription() << std::endl;
        exit(-1);
    }
}

#endif
