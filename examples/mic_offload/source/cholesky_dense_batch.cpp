/* file: cholesky_dense_batch.cpp */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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
!    C++ example of Cholesky decomposition
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-CHOLESKY_BATCH"></a>
 * \example cholesky_dense_batch.cpp
 */

#include "daal.h"
#include "service.h"

#include "offload.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;

/* Input data set parameters */
string datasetFileName = "../data/batch/cholesky.csv";

void serializeNumericTable(NumericTablePtr dataTable, byte **buffer, size_t *length);
NumericTablePtr deserializeNumericTable(byte *buffer, size_t size);
void serializeResult(services::SharedPtr<cholesky::Result> result, byte **buffer, size_t *length);
services::SharedPtr<cholesky::Result> deserializeResult(byte *buffer, size_t length);

int main(int argc, char *argv[])
{
    checkArguments(argc, argv, 1, &datasetFileName);

    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> dataSource(datasetFileName, DataSource::doAllocateNumericTable,
                                                 DataSource::doDictionaryFromContext);

    /* Retrieve the data from the input file */
    dataSource.loadDataBlock();


    byte *bufferin,*bufferout,*bufferout_card;
    size_t lengthin,lengthout;

    NumericTablePtr dataTable = dataSource.getNumericTable();
    serializeNumericTable(dataTable, &bufferin, &lengthin);

        _Offload_status x;
        OFFLOAD_STATUS_INIT(x);

    #pragma offload target(mic:0) status(x) in(bufferin:length(lengthin)) in(lengthin) out(lengthout) nocopy(bufferout_card)
    {
        if (_Offload_get_device_number() < 0) {
            printf("optional offload ran on CPU\n");
        } else {
            printf("optional offload ran on COPROCESSOR\n");
        }

        NumericTablePtr dataTable_card = deserializeNumericTable(bufferin, lengthin);

        /* Create an algorithm to compute Cholesky decomposition using the default method */
        cholesky::Batch<> algorithm;

        /* Set input objects for the algorithm */
        algorithm.input.set(cholesky::data, dataTable_card);

        /* Compute Cholesky decomposition */
        algorithm.compute();

        /* Get computed Cholesky decomposition */
        services::SharedPtr<cholesky::Result> res_card = algorithm.getResult();

        printNumericTable(res_card->get(cholesky::choleskyFactor));

        serializeResult(res_card, &bufferout_card, &lengthout);

    }
    if (x.result == OFFLOAD_SUCCESS) {
        printf("optional offload was successful\n");
    } else {
        printf("optional offload failed\n");
    }

    bufferout=(byte *)malloc(sizeof(byte)*lengthout);
    for(int i=0;i<lengthout;i++) bufferout[i]=0;

    #pragma offload target(mic:0) out(bufferout:length(lengthout)) nocopy(bufferout_card,lengthout)
    {
        for(int i=0;i<lengthout;i++) bufferout[i]=bufferout_card[i];
    }

    services::SharedPtr<cholesky::Result> res = deserializeResult(bufferout, lengthout);
    printNumericTable(res->get(cholesky::choleskyFactor));

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
    NumericTablePtr dataTable = NumericTablePtr( new HomogenNumericTable<float>() );

    /* Deserialize the numeric table from the data archive */
    dataTable->deserialize(dataArch);

    return dataTable;
}

void serializeResult(services::SharedPtr<cholesky::Result> result, byte **buffer, size_t *length)
{
    /* Create a data archive to serialize the numeric table */
    InputDataArchive dataArch;

    /* Serialize the numeric table into the data archive */
    result->serialize(dataArch);

    /* Get the length of the serialized data in bytes */
    *length = dataArch.getSizeOfArchive();

    /* Store the serialized data in an array */
    *buffer = new byte[*length];
    dataArch.copyArchiveToArray(*buffer, *length);
}

services::SharedPtr<cholesky::Result> deserializeResult(byte *buffer, size_t length)
{
    /* Create a data archive to deserialize the numeric table */
    OutputDataArchive dataArch(buffer, length);

    /* Create a numeric table object */
    services::SharedPtr<cholesky::Result> dataTable = services::SharedPtr<cholesky::Result>(new cholesky::Result);

    /* Deserialize the numeric table from the data archive */
    dataTable->deserialize(dataArch);

    return dataTable;
}
