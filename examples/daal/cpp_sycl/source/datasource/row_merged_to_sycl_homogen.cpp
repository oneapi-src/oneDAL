/* file: datastructures_usm.cpp */
/*******************************************************************************
 * Copyright 2021 Intel Corporation
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
!    Example of the use of Unified Shared Memory in Row Merged Numeric Table
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-ROWMERGEDTOSYCLHOMOGEN"></a>
 * \example datastructures_usm.cpp
 */

#include "daal_sycl.h"
#include "service_sycl.h"

using namespace daal;
using namespace daal::services;
using namespace daal::data_management;

using daal::data_management::internal::convertToSyclHomogen;
using daal::data_management::internal::SyclHomogenNumericTable;
using daal::services::internal::SyclExecutionContext;

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

/* Fill the buffer with pseudo random numbers generated with MinStd engine */
void generateData(float *dataBlock, size_t nRows, size_t nCols) {
  for (size_t i = 0; i < nRows; i++) {
    constexpr float genMax = 2147483647.0f;
    uint32_t genState = 7777 + i * i;
    genState = generateMinStd(genState);
    genState = generateMinStd(genState);
    for (size_t j = 0; j < nCols; j++) {
      dataBlock[i * nCols + j] = (float)genState / genMax;
      genState = generateMinStd(genState);
    }
  }
}

int main(int argc, char *argv[]) {
  constexpr size_t nCols = 10;
  constexpr size_t nRows = 10000;

  for (const auto &deviceDescriptor : getListOfDevices()) {
    const auto &device = deviceDescriptor.second;
    const auto &deviceName = deviceDescriptor.first;
    std::cout << "Running on " << deviceName << std::endl << std::endl;

    /* Crate SYCL* queue with desired device */
    cl::sycl::queue queue{device};

    /* Set the queue to default execution context */
    Environment::getInstance()->setDefaultExecutionContext(
        SyclExecutionContext{queue});

    SharedPtr<RowMergedNumericTable> mergedTable =
        RowMergedNumericTable::create();

    std::vector<float *> memChunks;
    bool allocatedAllBlocks = true;

    for (size_t i = 0; i < 3; i++) {
      /* Allocate memory on host to store input data */
      float *dataBlock = (float *)malloc(sizeof(float) * nRows * nCols);
      if (!dataBlock) {
        std::cout << "Allocation failed" << std::endl;
        allocatedAllBlocks = false;
        for (size_t i = 0; i < memChunks.size(); i++) {
          free(memChunks[i]);
        }
        break;
      }
      memChunks.push_back(dataBlock);

      /* Fill allocated memory block with generated numbers */
      generateData(dataBlock, nRows, nCols);

      /* Create numeric table from shared memory */
      NumericTablePtr dataTable =
          HomogenNumericTable<float>::create(dataBlock, nCols, nRows);

      /* Add to row merged table */
      mergedTable->addNumericTable(dataTable);
    }
    if (!allocatedAllBlocks) {
      continue;
    }

    /* Convert row merged table to sycl homogen one */
    Status st;
    NumericTablePtr tablePtr = convertToSyclHomogen<float>(*mergedTable, st);
    if (!st.ok()) {
      std::cout << "Failed to convert row merged table to SYCL homogen one"
                << std::endl;
      return -1;
    }

    /* Compute correlation matrix of generated dataset */
    NumericTablePtr covariance = computeCorrelationMatrix(tablePtr);

    /* Print the results */
    printNumericTable(covariance, "Covariance matrix:");

    /* Free data blocks*/
    for (size_t i = 0; i < memChunks.size(); i++) {
      free(memChunks[i]);
    }
  }

  return 0;
}
