/* file: cos_dist_dense_batch.cpp */
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
!    C++ example of computing a cosine distance matrix
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-COSINE_DISTANCE_BATCH"></a>
 * \example cos_dist_dense_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;

/* Input data set parameters */
const string datasetFileName = "../data/batch/distance.csv";

int main(int argc, char *argv[])
{
    checkArguments(argc, argv, 1, &datasetFileName);

    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> dataSource(datasetFileName, DataSource::doAllocateNumericTable,
                                                 DataSource::doDictionaryFromContext);

    /* Retrieve the data from the input file */
    dataSource.loadDataBlock();

    /* Create an algorithm to compute a cosine distance matrix using the default method */
    cosine_distance::Batch<> algorithm;

    /* Set input objects for the algorithm */
    algorithm.input.set(cosine_distance::data, dataSource.getNumericTable());

    /* Compute a cosine distance matrix */
    algorithm.compute();

    /* Get the computed cosine distance matrix */
    cosine_distance::ResultPtr res = algorithm.getResult();

    printNumericTable(res->get(cosine_distance::cosineDistance), "Cosine distance", 15);

    return 0;
}
