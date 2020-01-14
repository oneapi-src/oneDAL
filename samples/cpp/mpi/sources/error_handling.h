/* file: error_handling.h */
/*******************************************************************************
* Copyright 2017-2020 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/*
!  Content:
!    Auxiliary error-handling functions used in C++ samples
!******************************************************************************/

#ifndef _ERROR_HANDLING_H
#define _ERROR_HANDLING_H

const int fileError = -1001;

void checkAllocation(void * ptr)
{
    if (!ptr)
    {
        std::cout << "Error: Memory allocation failed" << std::endl;
        exit(-1);
    }
}

void checkPtr(void * ptr)
{
    if (!ptr)
    {
        std::cout << "Error: NULL pointer" << std::endl;
        exit(-2);
    }
}

void fileOpenError(const char * filename)
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

#endif
