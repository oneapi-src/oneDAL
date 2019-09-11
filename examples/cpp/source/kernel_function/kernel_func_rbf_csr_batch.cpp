/* file: kernel_func_rbf_csr_batch.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
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

using namespace std;
using namespace daal;
using namespace daal::algorithms;

/* Input data set parameters */
string leftDatasetFileName  = "../data/batch/kernel_function_csr.csv";
string rightDatasetFileName = "../data/batch/kernel_function_csr.csv";

/* Kernel algorithm parameters */
const double sigma = 1.0;       /* RBF kernel coefficient */

int main(int argc, char *argv[])
{
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
