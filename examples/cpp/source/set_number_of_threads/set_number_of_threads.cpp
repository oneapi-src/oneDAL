/* file: set_number_of_threads.cpp */
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
!    C++ example of setting the maximum number of threads
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-SET_NUMBER_OF_THREADS"></a>
 * \example set_number_of_threads.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;

/* Input data set parameters */
string datasetFileName     = "../data/batch/kmeans_dense.csv";

/* K-Means algorithm parameters */
const size_t nClusters   = 20;
const size_t nIterations = 5;
const size_t nThreads    = 2;
size_t nThreadsInit;
size_t nThreadsNew;

int main(int argc, char *argv[])
{
    checkArguments(argc, argv, 1, &datasetFileName);

    /* Get the number of threads that is used by the library by default */
    nThreadsInit = services::Environment::getInstance()->getNumberOfThreads();

    /* Set the maximum number of threads to be used by the library */
    services::Environment::getInstance()->setNumberOfThreads(nThreads);

    /* Get the number of threads that is used by the library after changing */
    nThreadsNew = services::Environment::getInstance()->getNumberOfThreads();

    /* Initialize FileDataSource to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> dataSource(datasetFileName, DataSource::doAllocateNumericTable,
                                                 DataSource::doDictionaryFromContext);

    /* Retrieve the data from the input file */
    dataSource.loadDataBlock();

    /* Get initial clusters for the K-Means algorithm */
    kmeans::init::Batch<float, kmeans::init::randomDense> init(nClusters);

    init.input.set(kmeans::init::data, dataSource.getNumericTable());
    init.compute();

    NumericTablePtr centroids = init.getResult()->get(kmeans::init::centroids);

    /* Create an algorithm object for the K-Means algorithm */
    kmeans::Batch<> algorithm(nClusters, nIterations);

    algorithm.input.set(kmeans::data,           dataSource.getNumericTable());
    algorithm.input.set(kmeans::inputCentroids, centroids);

    /* Run computations */
    algorithm.compute();

    cout << "Initial number of threads:        " << nThreadsInit << endl;
    cout << "Number of threads to set:         " << nThreads << endl;
    cout << "Number of threads after setting:  " << nThreadsNew  << endl;

    return 0;
}
