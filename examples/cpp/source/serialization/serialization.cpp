/* file: serialization.cpp */
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
!    C++ example of numeric table serialization
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-SERIALIZATION"></a>
 * \example serialization.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;

/* Input data set parameters */
const string datasetFileName = "../data/batch/serialization.csv";

void serializeNumericTable(NumericTablePtr dataTable, byte **buffer, size_t *length);
NumericTablePtr deserializeNumericTable(byte *buffer, size_t size);

int main(int argc, char *argv[])
{
    checkArguments(argc, argv, 1, &datasetFileName);

    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> dataSource(datasetFileName, DataSource::doAllocateNumericTable,
                                                 DataSource::doDictionaryFromContext);

    /* Retrieve the data from the input file */
    dataSource.loadDataBlock();

    /* Retrieve a numeric table */
    NumericTablePtr dataTable = dataSource.getNumericTable();

    /* Print the original data */
    printNumericTable(dataTable, "Data before serialization:");

    /* Serialize the numeric table into the memory buffer */
    byte *buffer;
    size_t length;
    serializeNumericTable(dataTable, &buffer, &length);

    /* Deserialize the numeric table from the memory buffer */
    NumericTablePtr restoredDataTable = deserializeNumericTable(buffer, length);

    /* Print the restored data */
    printNumericTable(restoredDataTable, "Data after deserialization:");

    delete [] buffer;
    return 0;
}

void serializeNumericTable(NumericTablePtr dataTable, byte **buffer, size_t *length)
{
    /* Create a data archive to serialize the numeric table */
    InputDataArchive dataArch;

    /* Serialize the numeric table into the data archive */
    dataTable->serialize(dataArch);

    /* Get the length of the serialized data in bytes */
    *length = dataArch.getSizeOfArchive();

    /* Store the serialized data in an array */
    *buffer = new byte[*length];
    dataArch.copyArchiveToArray(*buffer, *length);
}

NumericTablePtr deserializeNumericTable(byte *buffer, size_t length)
{
    /* Create a data archive to deserialize the numeric table */
    OutputDataArchive dataArch(buffer, length);

    /* Create a numeric table object */
    NumericTablePtr dataTable = NumericTablePtr( new HomogenNumericTable<>() );

    /* Deserialize the numeric table from the data archive */
    dataTable->deserialize(dataArch);

    return dataTable;
}
