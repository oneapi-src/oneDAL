/* file: datasource_kdb.cpp */
/*******************************************************************************
* Copyright 2017-2018 Intel Corporation.
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
! C++ sample of a Kdb data source.
!
! 1) Connect to the database and create a table there.
!
! 2) Read the data from it using the KDBDataSource functionality. Print the data from the table.
!
! 3) Delete the table from the database and disconnect.
!
!******************************************************************************/

#if defined(_WIN32) || defined(_WIN64)
#define NOMINMAX
#include <windows.h>
#endif

#define KXVER 3

#include <iostream>

#include "daal.h"
#include "service.h"
#include "data_management/data_source/kdb_data_source.h"

using namespace daal::algorithms;
using namespace daal;
using namespace std;

string dataSourceName = "daal-examples-kdb.inn.intel.com";
size_t dataSourcePort = 9999;
string dataSourceUsername = "";
string dataSourcePassword = "";

string tableName = "test_table";

int main(int argc, char *argv[])
{
    I handle=khpu(const_cast<char*>(dataSourceName.c_str()), dataSourcePort,
                  const_cast<char*>((dataSourceUsername + ":" + dataSourcePassword).c_str()));

    if (handle < 0)
    {
        std::cout << "Cannot connect" << std::endl;
        exit(-1);
    }

    if (handle == 0)
    {
        std::cout << "Wrong credentials" << std::endl;
        exit(-1);
    }

    K result = k(handle,
                 const_cast<char*>((tableName + ":([]x:1 2 3 4 5 6; y:2.1 3.45 -2.1 4.2 2.1 0.05; z:6 1 5 2 4 3; class:0 1 1 0 1 0)").c_str()),
                 (K)0);

    KDBDataSource<KDBFeatureManager> dataSource(
        dataSourceName, dataSourcePort, tableName, dataSourceUsername, dataSourcePassword,
        DataSource::doAllocateNumericTable, DataSource::doDictionaryFromContext);

    /* Get the number of rows in the data table */
    size_t nRows = dataSource.getNumberOfAvailableRows();

    printf ("number of rows = %d\n", (int)nRows);

    /* Load the number of rows from the data table */
    dataSource.loadDataBlock();

    /* Print the numeric table */
    printNumericTable(dataSource.getNumericTable());

    result = k(handle, const_cast<char*>(("delete " + tableName + " from `.").c_str()), (K)0);

    kclose(handle);

    return 0;
}
