/* file: kernel_func_rbf_csr_batch.cpp */
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
!    C++ example of computing a radial basis function (RBF) kernel
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-KERNEL_FUNCTION_RBF_CSR_BATCH"></a>
 * \example kernel_func_rbf_csr_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;

/* Input data set parameters */
std::string leftDatasetFileName = "../data/batch/kernel_function_csr.csv";
std::string rightDatasetFileName = "../data/batch/kernel_function_csr.csv";

/* Kernel algorithm parameters */
const double sigma = 1.0; /* RBF kernel coefficient */

int main(int argc, char* argv[]) {
    checkArguments(argc, argv, 1, &leftDatasetFileName);
    checkArguments(argc, argv, 1, &rightDatasetFileName);

    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    CSRNumericTablePtr leftData(createSparseTable<float>(leftDatasetFileName));
    CSRNumericTablePtr rightData(createSparseTable<float>(rightDatasetFileName));

    /* Create algorithm objects for the kernel algorithm using the default method */
    kernel_function::rbf::Batch<float, kernel_function::rbf::fastCSR> algorithm;

    /* Set the kernel algorithm parameter */
    algorithm.parameter.sigma = sigma;
    algorithm.parameter.computationMode = kernel_function::matrixMatrix;

    /* Set an input data table for the algorithm */
    algorithm.input.set(kernel_function::X, leftData);
    algorithm.input.set(kernel_function::Y, rightData);

    /* Compute the RBF kernel */
    algorithm.compute();

    /* Get the computed results */
    kernel_function::ResultPtr result = algorithm.getResult();

    /* Print the results */
    printNumericTable(result->get(kernel_function::values), "Values");

    return 0;
}
