/* file: datastructures_arrow.cpp */
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

int main(int argc, char * argv[])
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
    const arrow::Status makeReaderStatus =
        TableReader::Make(default_memory_pool(), file, ReadOptions::Defaults(), ParseOptions::Defaults(), ConvertOptions::Defaults(), &tableReader);
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
