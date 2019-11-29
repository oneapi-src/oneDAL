/* file: low_order_moms_dense_batch.cpp */
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
!    C++ example of computing low order moments in the batch
!    processing mode with DPC++ interfaces
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-LOW_ORDER_MOMENTS_DENSE_BATCH">
 * \example low_order_moms_dense_batch.cpp
 */

#include "daal_sycl.h"
#include "service.h"
#include "service_sycl.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;

/* Input data set parameters */
string datasetFileName = "../data/batch/covcormoments_dense.csv";

void printResults(const low_order_moments::ResultPtr & res);

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

        FileDataSource<CSVFeatureManager> dataSource(datasetFileName, DataSource::notAllocateNumericTable, DataSource::doDictionaryFromContext);

        auto data = SyclHomogenNumericTable<>::create(10, 0, NumericTable::notAllocate);
        dataSource.loadDataBlock(data.get());

        /* Create an algorithm to compute low order moments using the default method */
        low_order_moments::Batch<> algorithm;
        /* Set input objects for the algorithm */
        algorithm.input.set(low_order_moments::data, data);

        /* Compute low order moments */
        algorithm.compute();

        /* Get the computed low order moments */
        low_order_moments::ResultPtr res = algorithm.getResult();

        printResults(res);
    }

    return 0;
}

void printResults(const low_order_moments::ResultPtr & res)
{
    printNumericTable(res->get(low_order_moments::minimum), "Minimum:");
    printNumericTable(res->get(low_order_moments::maximum), "Maximum:");
    printNumericTable(res->get(low_order_moments::sum), "Sum:");
    printNumericTable(res->get(low_order_moments::sumSquares), "Sum of squares:");
    printNumericTable(res->get(low_order_moments::sumSquaresCentered), "Sum of squared difference from the means:");
    printNumericTable(res->get(low_order_moments::mean), "Mean:");
    printNumericTable(res->get(low_order_moments::secondOrderRawMoment), "Second order raw moment:");
    printNumericTable(res->get(low_order_moments::variance), "Variance:");
    printNumericTable(res->get(low_order_moments::standardDeviation), "Standard deviation:");
    printNumericTable(res->get(low_order_moments::variation), "Variation:");
}
