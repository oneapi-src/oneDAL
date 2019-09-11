/* file: kmeans_csr_batch.cpp */
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
!    C++ example of sparse K-Means clustering in the batch processing mode
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-KMEANS_CSR_BATCH"></a>
 * \example kmeans_csr_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;

typedef float algorithmFPType; /* Algorithm floating-point type */

/* Input data set parameters */
string datasetFileName     = "../data/batch/kmeans_csr.csv";

/* K-Means algorithm parameters */
const size_t nClusters   = 20;
const size_t nIterations = 5;

int main(int argc, char *argv[])
{
    checkArguments(argc, argv, 1, &datasetFileName);

    /* Retrieve the data from the input file */
    CSRNumericTablePtr dataTable(createSparseTable<float>(datasetFileName));

    /* Get initial clusters for the K-Means algorithm */
    kmeans::init::Batch<algorithmFPType, kmeans::init::randomCSR> init(nClusters);

    init.input.set(kmeans::init::data, dataTable);
    init.compute();

    NumericTablePtr centroids = init.getResult()->get(kmeans::init::centroids);

    /* Create an algorithm object for the K-Means algorithm */
    kmeans::Batch<algorithmFPType, kmeans::lloydCSR> algorithm(nClusters, nIterations);

    algorithm.input.set(kmeans::data,           dataTable);
    algorithm.input.set(kmeans::inputCentroids, centroids);

    algorithm.compute();

    /* Print the clusterization results */
    printNumericTable(algorithm.getResult()->get(kmeans::assignments), "First 10 cluster assignments:", 10);
    printNumericTable(algorithm.getResult()->get(kmeans::centroids  ), "First 10 dimensions of centroids:", 20, 10);
    printNumericTable(algorithm.getResult()->get(kmeans::objectiveFunction), "Objective function value:");

    return 0;
}
