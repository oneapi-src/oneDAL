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
#include "service.h"

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
    covAlg.input.set(covariance::data, dataTable);
    covAlg.parameter.outputMatrixType = covariance::correlationMatrix;
    covAlg.compute();

    return covAlg.getResult()->get(covariance::correlation);
}

int main(int argc, char *argv[])
{
    constexpr size_t nRows = 100000;
    constexpr size_t nCols = 5;

    cl::sycl::queue queue{cl::sycl::gpu_selector{}};
    Environment::getInstance()->setDefaultExecutionContext(
        SyclExecutionContext{queue}
    );

    float *dataDevice = (float *)cl::sycl::malloc_device(
        sizeof(float) * nRows * nCols, queue.get_device(), queue.get_context());

    cl::sycl::event event = generateData(queue, dataDevice, nRows, nCols);

    NumericTablePtr dataTable = SyclHomogenNumericTable<float>::create(
        dataDevice, cl::sycl::usm::alloc::device, nCols, nRows, event);

    NumericTablePtr covariance = computeCorrelationMatrix(dataTable);

    printNumericTable(covariance, "Covariance matrix:");

    // {
    //     float *dataHost = (float *)cl::sycl::malloc_host(sizeof(float) * nRows * nCols,
    //                                                      queue.get_context());
    //     queue.memcpy(dataHost, dataDevice, sizeof(float) * nRows * nCols).wait();
    //     for (size_t i = 0; i < nRows; i++)
    //     {
    //         for (size_t j = 0; j < nCols; j++)
    //         {
    //             std::cout << dataHost[i * nCols + j] << " ";
    //         }
    //         std::cout << std::endl;
    //     }
    //     cl::sycl::free(dataHost, queue.get_context());
    // }

    cl::sycl::free(dataDevice, queue.get_context());

    // for (const auto& device : { cl::sycl::gpu_selector() })
    // {
    //     // const auto& nameDevice = deviceSelector.first;
    //     // const auto& device = deviceSelector.second;
    //     // std::cout << "Running on " << nameDevice << "\n\n";

    //     daal::services::SyclExecutionContext ctx(queue);
    //     services::Environment::getInstance()->setDefaultExecutionContext(ctx);

    //     FileDataSource<CSVFeatureManager> dataSource(datasetFileName,
    //                                                  DataSource::doAllocateNumericTable,
    //                                                  DataSource::doDictionaryFromContext);
    //     dataSource.loadDataBlock();
    //     const auto data = dataSource.getNumericTable();

    //     const size_t dataSize = data->getNumberOfRows() * data->getNumberOfColumns();

    //     {
    //         BlockDescriptor<> block;
    //         data->getBlockOfRows(0, data->getNumberOfRows(), readOnly, block);
    //         queue.memcpy(dataDevice, block.getBlockPtr(), sizeof(float) * dataSize).wait();
    //         data->releaseBlockOfRows(block);
    //     }

    //     const auto dataTable = data_management::SyclHomogenNumericTable<float>::create(
    //         dataDevice, cl::sycl::usm::alloc::device,
    //         data->getNumberOfColumns(), data->getNumberOfRows());

    //     covariance::Batch<> algorithm;
    //     algorithm.input.set(covariance::data, dataTable);

    //     algorithm.parameter.outputMatrixType = covariance::correlationMatrix;

    //     algorithm.compute();
    //     algorithm.compute();



    // }
    return 0;
}
