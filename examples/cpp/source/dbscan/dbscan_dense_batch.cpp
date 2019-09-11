/* file: dbscan_dense_batch.cpp */
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
!    C++ example of dense DBSCAN clustering in the batch processing mode
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-DBSCAN_BATCH"></a>
 * \example dbscan_dense_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;

/* Input data set parameters */
string datasetFileName     = "../data/batch/dbscan_dense.csv";

/* DBSCAN algorithm parameters */
const float epsilon = 0.02f;
const size_t minObservations = 180;

int main(int argc, char *argv[])
{
    checkArguments(argc, argv, 1, &datasetFileName);

    /* Initialize FileDataSource to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> dataSource(datasetFileName, DataSource::doAllocateNumericTable,
                                                 DataSource::doDictionaryFromContext);

    /* Retrieve the data from the input file */
    dataSource.loadDataBlock();

    /* Create an algorithm object for the DBSCAN algorithm */
    dbscan::Batch<> algorithm(epsilon, minObservations);

    algorithm.input.set(dbscan::data, dataSource.getNumericTable());

    algorithm.compute();

    /* Print the clusterization results */
    printNumericTable(algorithm.getResult()->get(dbscan::nClusters), "Number of clusters:");
    printNumericTable(algorithm.getResult()->get(dbscan::assignments), "Assignments of first 20 observations:", 20);

    return 0;
}
