/* file: sorting_dense_batch.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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
!    C++ example of sorting the observations matrix
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-SORTING_BATCH"></a>
 * \example sorting_dense_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;
using namespace std;

/* Input data set parameters */
string datasetFileName = "../data/batch/sorting.csv";

int main(int argc, char *argv[])
{
    checkArguments(argc, argv, 1, &datasetFileName);

    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> dataSource(datasetFileName, DataSource::doAllocateNumericTable, DataSource::doDictionaryFromContext);

    /* Retrieve the data from the input file */
    dataSource.loadDataBlock();

    /* Create algorithm objects to sort data using the default (radix) method */
    sorting::Batch<> algorithm;

    /* Print the input observations matrix */
    printNumericTable(dataSource.getNumericTable(), "Initial matrix of observations:");

    /* Set input objects for the algorithm */
    algorithm.input.set(sorting::data, dataSource.getNumericTable());

    /* Sort data observations */
    algorithm.compute();

    /* Get the sorting result */
    sorting::ResultPtr res = algorithm.getResult();

    printNumericTable(res->get(sorting::sortedData), "Sorted matrix of observations:");

    return 0;
}
