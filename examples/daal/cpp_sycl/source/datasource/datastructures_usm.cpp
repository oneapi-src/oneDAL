/* file: datastructures_usm.cpp */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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
!    Example of the use of Unified Shared Memory in Numeric Table
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-DATASTRUCTURES_USM"></a>
 * \example datastructures_usm.cpp
 */

#include "daal_sycl.h"
#include "service_sycl.h"

using namespace daal;
using namespace daal::services;
using namespace daal::data_management;

using daal::services::internal::SyclExecutionContext;
using daal::data_management::internal::SyclHomogenNumericTable;

#if (defined(__SYCL_COMPILER_VERSION)) && (__SYCL_COMPILER_VERSION >= 20191001)
#define IS_USM_SUPPORTED
#endif

uint32_t generateMinStd(uint32_t x) {
    constexpr uint32_t a = 16807;
    constexpr uint32_t c = 0;
    constexpr uint32_t m = 2147483647;
    return (a * x + c) % m;
}

/* Compute correlation matrix */
NumericTablePtr computeCorrelationMatrix(const NumericTablePtr &table) {
    using namespace daal::algorithms;

    covariance::Batch<> covAlg;
    covAlg.input.set(covariance::data, table);
    covAlg.parameter.outputMatrixType = covariance::correlationMatrix;
    covAlg.compute();

    return covAlg.getResult()->get(covariance::correlation);
}

/* Detect wether USM extensions are supported */
#ifdef IS_USM_SUPPORTED

/* Fill the buffer with pseudo random numbers generated with MinStd engine */
sycl::event generateData(sycl::queue &q, float *deviceData, size_t nRows, size_t nCols) {
    using namespace cl::sycl;
    return q.submit([&](handler &cgh) {
        cgh.parallel_for<class FillTable>(range<1>(nRows), [=](id<1> idx) {
            constexpr float genMax = 2147483647.0f;
            uint32_t genState = 7777 + idx[0] * idx[0];
            genState = generateMinStd(genState);
            genState = generateMinStd(genState);
            for (size_t j = 0; j < nCols; j++) {
                deviceData[idx[0] * nCols + j] = (float)genState / genMax;
                genState = generateMinStd(genState);
            }
        });
    });
}

int main(int argc, char *argv[]) {
    constexpr size_t nCols = 10;
    constexpr size_t nRows = 10000;

    for (const auto &deviceDescriptor : getListOfDevices()) {
        const auto &device = deviceDescriptor.second;

        const auto &deviceName = deviceDescriptor.first;
        std::cout << "Running on " << deviceName << std::endl << std::endl;

        /* Crate SYCL* queue with desired device */
        sycl::queue queue{ device };

        /* Set the queue to default execution context */
        Environment::getInstance()->setDefaultExecutionContext(SyclExecutionContext{ queue });

        /* Allocate shared memory to store input data */
        float *dataDevice = (float *)sycl::malloc_shared(sizeof(float) * nRows * nCols,
                                                         queue.get_device(),
                                                         queue.get_context());
        if (!dataDevice) {
            std::cout << "USM allocation failed on " << deviceName << std::endl;
            continue;
        }

        /* Fill allocated memory block with generated numbers */
        generateData(queue, dataDevice, nRows, nCols).wait();

        /* Create numeric table from shared memory */
        NumericTablePtr dataTable =
            SyclHomogenNumericTable<float>::create(dataDevice, nCols, nRows, queue);

        /* Compute correlation matrix of generated dataset */
        NumericTablePtr covariance = computeCorrelationMatrix(dataTable);

        /* Print the results */
        printNumericTable(covariance, "Covariance matrix:");

        /* Free USM data */
        sycl::free(dataDevice, queue.get_context());
    }

    return 0;
}

#else /* USM is not supported */

int main(int argc, char *argv[]) {
    std::cout << "USM extensions are not available, make sure "
              << "the compiler and runtime support USM" << std::endl;
    return 0;
}

#endif // IS_USM_SUPPORTED
