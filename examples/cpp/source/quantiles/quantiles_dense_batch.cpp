/* file: quantiles_dense_batch.cpp */
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
!    C++ example of computing quantiles
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-QUANTILES_BATCH"></a>
 * \example quantiles_dense_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;
using namespace std;

/* Input data set parameters */
string datasetFileName = "../data/batch/quantiles.csv";

int main(int argc, char *argv[])
{
    checkArguments(argc, argv, 1, &datasetFileName);

    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> dataSource(datasetFileName, DataSource::doAllocateNumericTable, DataSource::doDictionaryFromContext);

    /* Retrieve the data from the input file */
    dataSource.loadDataBlock();

    /* Create an algorithm to compute quantiles using the default method */
    quantiles::Batch<> algorithm;

    /* Set input objects for the algorithm */
    algorithm.input.set(quantiles::data, dataSource.getNumericTable());

    /* Compute quantiles */
    algorithm.compute();

    /* Get the computed quantiles */
    quantiles::ResultPtr res = algorithm.getResult();

    printNumericTable(res->get(quantiles::quantiles), "Quantiles");

    return 0;
}
