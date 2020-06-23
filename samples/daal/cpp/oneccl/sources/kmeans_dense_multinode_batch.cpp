/* file: kmeans_dense_multinode_batch.cpp */
/*******************************************************************************
* Copyright 2020 Intel Corporation
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
!    C++ sample of dense K-Means clustering in the multinode batch processing mode
!    with DPC++ interfaces
!******************************************************************************/

/**
 * <a name="DAAL-SAMPLE-CPP-KMEANS_DENSE_MULTINODE_BATCH"></a>
 * \sample kmeans_dense_multinode_batch.cpp
 */

#include "daal_sycl.h"
#include "service.h"
#include "service_sycl.h"
#include "services/comm_detect.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;

/* Input data set parameters */
string datasetFileName = "./data/kmeans_dense.csv";

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
        if (!device.is_gpu()) continue;
        cl::sycl::queue queue(device);
        std::cout << "Running on " << nameDevice << "\n\n";

        daal::services::SyclExecutionContext ctx(queue);
        services::Environment::getInstance()->setDefaultExecutionContext(ctx);

        daal::preview::services::OneCclCommunicator comm(queue);
        daal::preview::services::CommManager::getInstance()->setDefaultCommunicator(comm);

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

        algorithms::preview::kmeans::MultiNodeBatch<> algorithm(nClusters, nIterations);

        algorithm.input.set(kmeans::data, data);
        algorithm.input.set(kmeans::inputCentroids, centroids);

        algorithm.compute();

        /* Print the clusterization results */
        printNumericTable(((kmeans::Result *)algorithm.getResult().get())->get(kmeans::assignments), "First 10 cluster assignments:", 10);
        printNumericTable(((kmeans::Result *)algorithm.getResult().get())->get(kmeans::centroids), "First 10 dimensions of centroids:", 20, 10);
        printNumericTable(((kmeans::Result *)algorithm.getResult().get())->get(kmeans::objectiveFunction), "Objective function value:");
    }

    return 0;
}
