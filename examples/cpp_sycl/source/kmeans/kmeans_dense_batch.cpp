/* file: kmeans_dense_batch.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
!    C++ example of dense K-Means clustering in the batch processing mode
!    with DPC++ interfaces
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-KMEANS_DENSE_BATCH"></a>
 * \example kmeans_dense_batch.cpp
 */

#include "daal_sycl.h"
#include "service.h"
#include "service_sycl.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;

/* Input data set parameters */
string datasetFileName = "../data/batch/kmeans_dense.csv";

/* K-Means algorithm parameters */
const size_t nFeatures   = 20; /* Number of features in input data sets */
const size_t nClusters   = 20;
const size_t nIterations = 5;

int main(int argc, char * argv[])
{
    checkArguments(argc, argv, 1, &datasetFileName);

    for (const auto & deviceSelector : getListOfDevices())
    {
        const auto & nameDevice = deviceSelector.first;
        const auto & device     = deviceSelector.second;
        cl::sycl::queue queue(device);
        std::cout << "Running on " << nameDevice << "\n\n";

        daal::services::SyclExecutionContext ctx(queue);
        services::Environment::getInstance()->setDefaultExecutionContext(ctx);

        /* Initialize FileDataSource to retrieve the input data from a .csv file */
        FileDataSource<CSVFeatureManager> dataSource(datasetFileName, DataSource::notAllocateNumericTable, DataSource::doDictionaryFromContext);

        NumericTablePtr data = SyclHomogenNumericTable<>::create(nFeatures, 0, NumericTable::notAllocate);

        /* Retrieve the data from the input file */
        dataSource.loadDataBlock(data.get());

        /* Get initial clusters for the K-Means algorithm */
        kmeans::init::Batch<float, kmeans::init::randomDense> init(nClusters);

        init.input.set(kmeans::init::data, data);
        init.compute();

        NumericTablePtr centroids = init.getResult()->get(kmeans::init::centroids);

        kmeans::Batch<> algorithm(nClusters, nIterations);

        algorithm.input.set(kmeans::data, data);
        algorithm.input.set(kmeans::inputCentroids, centroids);

        algorithm.compute();

        /* Print the clusterization results */
        printNumericTable(algorithm.getResult()->get(kmeans::assignments), "First 10 cluster assignments:", 10);
        printNumericTable(algorithm.getResult()->get(kmeans::centroids), "First 10 dimensions of centroids:", 20, 10);
        printNumericTable(algorithm.getResult()->get(kmeans::objectiveFunction), "Objective function value:");
    }

    return 0;
}
