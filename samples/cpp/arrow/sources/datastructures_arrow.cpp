/* file: datastructures_arrow.cpp */
/*******************************************************************************
* Copyright 2017-2019 Intel Corporation.
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
*
* License:
* http://software.intel.com/en-us/articles/intel-sample-source-code-license-agr
* eement/
*******************************************************************************/

/*
! Content:
!
! C++ sample of a Apache Arrow Numeric Table.
!
! 1) Read CSV file and create a Apache Arrow table there.
!
! 2) Create the numeric table for Apache Arrow table. Print the data from the table.
!
!******************************************************************************/

#if defined(_WIN32) || defined(_WIN64)
#define NOMINMAX
#include <windows.h>
#endif

#include <iostream>

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/csv/api.h>

#include "daal.h"
#include "service.h"
#include "data_management/data/arrow_numeric_table.h"

using namespace daal::algorithms;
using namespace daal::services;
using namespace daal;
using namespace std;
using namespace arrow;
using namespace arrow::io;
using namespace arrow::csv;

const string csvFileName = "./data/datastructures_arrow.csv";

int main(int argc, char *argv[])
{
    /* Open CSV file */
    shared_ptr<ReadableFile> file;
    const arrow::Status openStatus = ReadableFile::Open(csvFileName, &file);
    if (!openStatus.ok())
    {
        cout << "Cannot open CSV file: " << openStatus.message() << endl;
        exit(-1);
    }

    /* Make the table reader */
    shared_ptr<TableReader> tableReader;
    const arrow::Status makeReaderStatus = TableReader::Make(default_memory_pool(), file, ReadOptions::Defaults(), ParseOptions::Defaults(),
                                                      ConvertOptions::Defaults(), &tableReader);
    if (!makeReaderStatus.ok())
    {
        cout << "Cannot make table reader: " << makeReaderStatus.message() << endl;
        exit(-1);
    }

    /* Read the table */
    shared_ptr<Table> table;
    const arrow::Status readTableStatus = tableReader->Read(&table);
    if (!readTableStatus.ok())
    {
        cout << "Cannot read table: " << readTableStatus.message() << endl;
        exit(-1);
    }

    /* Create the numeric table for Apache Arrow table */
    SharedPtr<ArrowImmutableNumericTable> nt = ArrowImmutableNumericTable::create(table);

    /* Print the numeric table */
    printNumericTable(nt);

    return 0;
}
