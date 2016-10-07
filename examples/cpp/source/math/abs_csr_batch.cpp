/* file: abs_csr_batch.cpp */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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
!    C++ example of abs algorithm.
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-ABS_CSR_BATCH"></a>
 * \example abs_csr_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;
using namespace daal::algorithms::math;

/* Input data set parameters */
string datasetName = "../data/batch/covcormoments_csr.csv";

int main(int argc, char *argv[])
{
    checkArguments(argc, argv, 1, &datasetName);

    /* Read datasetFileName from a file and create a numeric table to store input data */
    services::SharedPtr<CSRNumericTable> dataTable(createSparseTable<float>(datasetName));

    /* Create an algorithm */
    abs::Batch<float, abs::fastCSR> abs;

    /* Set an input object for the algorithm */
    abs.input.set(abs::data, dataTable);

    /* Compute Abs function */
    abs.compute();

    /* Print the results of the algorithm */
    services::SharedPtr<abs::Result> res = abs.getResult();
    printNumericTable(res->get(abs::value), "Abs result (first 5 rows):", 5);

    return 0;
}
