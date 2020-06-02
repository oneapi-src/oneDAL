/* file: pca_cor_dense_online.cpp */
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
!    C++ example of principal component analysis (PCA) using the correlation
!    method in the batch processing mode with DPC++ interfaces
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-PCA_CORRELATION_DENSE_ONLINE"></a>
 * \example pca_cor_dense_online.cpp
 */

#include "daal_sycl.h"
#include "service.h"
#include "service_sycl.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;

/* Input data set parameters */
const size_t nVectorsInBlock = 250;
const string dataFileName    = "../data/batch/pca_normalized.csv";

int main(int argc, char * argv[])
{
    checkArguments(argc, argv, 1, &dataFileName);

    for (const auto & deviceSelector : getListOfDevices())
    {
        const auto & nameDevice = deviceSelector.first;
        const auto & device     = deviceSelector.second;
        cl::sycl::queue queue(device);
        std::cout << "Running on " << nameDevice << "\n\n";

        daal::services::SyclExecutionContext ctx(queue);
        services::Environment::getInstance()->setDefaultExecutionContext(ctx);

        /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
        FileDataSource<CSVFeatureManager> dataSource(dataFileName, DataSource::notAllocateNumericTable, DataSource::doDictionaryFromContext);

        auto data = SyclHomogenNumericTable<>::create(10, 0, NumericTable::notAllocate);

        /* Create an algorithm for principal component analysis using the correlation method */
        pca::Online<> algorithm;

        /* Set the algorithm input data */
        while (dataSource.loadDataBlock(nVectorsInBlock, data.get()) == nVectorsInBlock)
        {
            /* Set input objects for the algorithm */
            algorithm.input.set(pca::data, data);

            /* Compute partial estimates */
            algorithm.compute();
        }

        algorithm.finalizeCompute();

        /* Print the results */
        pca::ResultPtr result = algorithm.getResult();

        printNumericTable(result->get(pca::eigenvalues), "Eigenvalues:");
        printNumericTable(result->get(pca::eigenvectors), "Eigenvectors:");
    }

    return 0;
}
