/* file: kernel_func_lin_dense_batch.cpp */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
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
!    C++ example of computing a linear kernel function
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-KERNEL_FUNCTION_LINEAR_DENSE_BATCH"></a>
 * \example kernel_func_lin_dense_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;

/* Input data set parameters */
string leftDatasetFileName  = "../data/batch/kernel_function.csv";
string rightDatasetFileName = "../data/batch/kernel_function.csv";

/* Kernel algorithm parameters */
const double k = 1.0;            /* Linear kernel coefficient in the k(X,Y) + b model */
const double b = 0.0;            /* Linear kernel coefficient in the k(X,Y) + b model */

int main(int argc, char *argv[])
{
    checkArguments(argc, argv, 1, &leftDatasetFileName);
    checkArguments(argc, argv, 1, &rightDatasetFileName);

    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> leftDataSource(leftDatasetFileName, DataSource::doAllocateNumericTable,
                                                     DataSource::doDictionaryFromContext);

    FileDataSource<CSVFeatureManager> rightDataSource(rightDatasetFileName, DataSource::doAllocateNumericTable,
                                                      DataSource::doDictionaryFromContext);

    /* Retrieve the data from the input file */
    leftDataSource.loadDataBlock();
    rightDataSource.loadDataBlock();

    /* Create algorithm objects for the kernel algorithm using the default method */
    kernel_function::linear::Batch<> algorithm;

    /* Set the kernel algorithm parameter */
    algorithm.parameter.k = k;
    algorithm.parameter.b = b;
    algorithm.parameter.computationMode = kernel_function::matrixMatrix;

    /* Set an input data table for the algorithm */
    algorithm.input.set(kernel_function::X, leftDataSource.getNumericTable());
    algorithm.input.set(kernel_function::Y, rightDataSource.getNumericTable());

    /* Compute the linear kernel function */
    algorithm.compute();

    /* Get the computed results */
    kernel_function::ResultPtr result = algorithm.getResult();

    /* Print the results */
    printNumericTable(result->get(kernel_function::values), "Values");

    return 0;
}
