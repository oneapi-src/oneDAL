/* file: utils.h */
/*******************************************************************************
* Copyright 2017-2019 Intel Corporation
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
!    Auxiliary functions used in C++ MySQL sample
!******************************************************************************/

#ifndef __UTILS_H__
#define __UTILS_H__

#include <ctime>
#include <string>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <algorithm>

#include "daal.h"

namespace utils
{
inline std::string trim(std::string & str)
{
    const char * ignore = " \n\t\r";
    str.erase(0, str.find_first_not_of(ignore));
    str.erase(str.find_last_not_of(ignore) + 1);
    return str;
}

inline std::string generateTableName(const std::string & connectionId)
{
    std::stringstream ss;

    ss << "daal_table"
       << "_" << connectionId << "_" << time(NULL);

    return ss.str();
}

inline void printHelp()
{
    std::cout << "Usage example:\n"
              << "  datasource_mysql.exe <connection_string>\n"
              << "\n"
              << "For more information about <connection_string> see\n"
              << "https://dev.mysql.com/doc/connector-odbc/en/connector-odbc-configuration-connection-without-dsn.html\n";
}

template <typename T>
inline void printArray(T * array, const size_t nPrintedCols, const size_t nPrintedRows, const size_t nCols, std::string message, size_t interval = 10)
{
    std::cout << std::setiosflags(std::ios::left);
    std::cout << message << std::endl;
    for (size_t i = 0; i < nPrintedRows; i++)
    {
        for (size_t j = 0; j < nPrintedCols; j++)
        {
            std::cout << std::setw(interval) << std::setiosflags(std::ios::fixed) << std::setprecision(3);
            std::cout << array[i * nCols + j];
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

inline void printNumericTable(const daal::data_management::NumericTablePtr & dataTable, const char * message = "", size_t nPrintedRows = 0,
                              size_t nPrintedCols = 0, size_t interval = 10)
{
    using namespace daal::data_management;

    size_t nRows = dataTable->getNumberOfRows();
    size_t nCols = dataTable->getNumberOfColumns();

    if (nPrintedRows != 0)
    {
        nPrintedRows = (std::min)(nRows, nPrintedRows);
    }
    else
    {
        nPrintedRows = nRows;
    }

    if (nPrintedCols != 0)
    {
        nPrintedCols = (std::min)(nCols, nPrintedCols);
    }
    else
    {
        nPrintedCols = nCols;
    }

    BlockDescriptor<DAAL_DATA_TYPE> block;
    {
        dataTable->getBlockOfRows(0, nRows, readOnly, block);
        printArray<DAAL_DATA_TYPE>(block.getBlockPtr(), nPrintedCols, nPrintedRows, nCols, message, interval);
    }
    dataTable->releaseBlockOfRows(block);
}

} // namespace utils

#endif
