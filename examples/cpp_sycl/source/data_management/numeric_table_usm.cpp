/* file: numeric_table_usm.cpp */
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
!    processing mode with DPC++ interfaces
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-NUMERIC_TABLE_USM"></a>
 * \example numeric_table_usm.cpp
 */

#include "daal_sycl.h"
#include "service_sycl.h"

using namespace daal;
using namespace daal::services;
using namespace daal::data_management;

uint32_t generateMinStd(uint32_t x)
{
    constexpr uint32_t a = 16807;
    constexpr uint32_t c = 0;
    constexpr uint32_t m = 2147483647;
    return (a * x + c) % m;
}

cl::sycl::event generateData(cl::sycl::queue &q, float *deviceData,
                             size_t nRows, size_t nCols)
{
    using namespace cl::sycl;
    return q.submit([&](handler &cgh)
    {
        cgh.parallel_for<class FillTable>(range<1>(nRows), [=](id<1> idx)
        {
            constexpr float genMax = 2147483647.0f;
            uint32_t genState = 7777 + idx[0] * idx[0];
            genState = generateMinStd(genState);
            genState = generateMinStd(genState);
            for (size_t j = 0; j < nCols; j++)
            {
                deviceData[idx[0] * nCols + j] = (float)genState / genMax;
                genState = generateMinStd(genState);
            }
        });
    });
}

NumericTablePtr computeCorrelationMatrix(const NumericTablePtr &table)
{
    using namespace daal::algorithms;

    covariance::Batch<> covAlg;
    covAlg.input.set(covariance::data, table);
    covAlg.parameter.outputMatrixType = covariance::correlationMatrix;
    covAlg.compute();

    return covAlg.getResult()->get(covariance::correlation);
}

int main(int argc, char *argv[])
{
    constexpr size_t nCols = 10;
    constexpr size_t nRows = 10000;

    for (const auto &deviceDescriptor : getListOfDevices())
    {
        const auto &device = deviceDescriptor.second;
        if (device.is_host())
        {
            /* Shared memory allocations do not work on host */
            continue;
        }

        const auto &deviceName = deviceDescriptor.first;
        std::cout << "Running on " << deviceName << std::endl << std::endl;

        cl::sycl::queue queue{device};

        Environment::getInstance()->setDefaultExecutionContext(
            SyclExecutionContext{queue}
        );

        float *dataDevice = (float *)cl::sycl::malloc_shared(
            sizeof(float) * nRows * nCols, queue.get_device(), queue.get_context());

        generateData(queue, dataDevice, nRows, nCols).wait();

        NumericTablePtr dataTable = SyclHomogenNumericTable<float>::create(
            dataDevice, cl::sycl::usm::alloc::shared, nCols, nRows);

        NumericTablePtr covariance = computeCorrelationMatrix(dataTable);

        printNumericTable(covariance, "Covariance matrix:");

        cl::sycl::free(dataDevice, queue.get_context());
    }

    return 0;
}
