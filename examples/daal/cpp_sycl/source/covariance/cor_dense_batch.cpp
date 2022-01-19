/* file: cor_dense_batch.cpp */
/*******************************************************************************
* Copyright 2014-2022 Intel Corporation
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
!    processing mode with DPC++ interfaces
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-CORRELATION_DENSE_BATCH"></a>
 * \example cor_dense_batch.cpp
 */

#include "daal_sycl.h"
#include "service.h"
#include "service_sycl.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;

using daal::services::internal::SyclExecutionContext;
using daal::data_management::internal::SyclHomogenNumericTable;

/* Input data set parameters */
const string datasetFileName = "../data/batch/covcormoments_dense.csv";

int main(int argc, char * argv[])
{
    checkArguments(argc, argv, 1, &datasetFileName);

    for (const auto & deviceSelector : getListOfDevices())
    {
        const auto & nameDevice = deviceSelector.first;
        const auto & device     = deviceSelector.second;
        cl::sycl::queue queue(device);
        std::cout << "Running on " << nameDevice << "\n\n";

        SyclExecutionContext ctx(queue);
        services::Environment::getInstance()->setDefaultExecutionContext(ctx);

        FileDataSource<CSVFeatureManager> dataSource(datasetFileName, DataSource::notAllocateNumericTable, DataSource::doDictionaryFromContext);

        auto data = SyclHomogenNumericTable<>::create(10, 0, NumericTable::notAllocate);
        dataSource.loadDataBlock(data.get());

        covariance::Batch<> algorithm;
        algorithm.input.set(covariance::data, data);

        algorithm.parameter.outputMatrixType = covariance::correlationMatrix;

        algorithm.compute();

        covariance::ResultPtr res = algorithm.getResult();

        printNumericTable(res->get(covariance::correlation), "Correlation matrix:");
        printNumericTable(res->get(covariance::mean), "Mean vector:");
    }
    return 0;
}
