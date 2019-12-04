/* file: cor_dense_online.cpp */
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
!    C++ example of dense correlation matrix computation in the batch
!    processing mode
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-CORRELATION_DENSE_ONLINE"></a>
 * \example cor_dense_online.cpp
 */

#include "daal_sycl.h"
#include "service.h"
#include "service_sycl.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;

/* Input data set parameters */
const string datasetFileName = "../data/batch/covcormoments_dense.csv";
const size_t nObservations   = 50;

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

        auto data = SyclHomogenNumericTable<>::create(10, 0, NumericTable::notAllocate);

        FileDataSource<CSVFeatureManager> dataSource(datasetFileName, DataSource::doAllocateNumericTable, DataSource::doDictionaryFromContext);

        covariance::Online<> algorithm;
        algorithm.parameter.outputMatrixType = covariance::correlationMatrix;

        while (dataSource.loadDataBlock(nObservations, data.get()) == nObservations)
        {
            /* Set input objects for the algorithm */
            algorithm.input.set(covariance::data, data);

            /* Compute partial estimates */
            algorithm.compute();
        }

        algorithm.finalizeCompute();

        covariance::ResultPtr res = algorithm.getResult();

        printNumericTable(res->get(covariance::covariance), "Covariance matrix:");
        printNumericTable(res->get(covariance::mean), "Mean vector:");
    }
    return 0;
}
