/* file: low_order_moms_csr_batch.cpp */
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
!    C++ example of computing low order moments in the batch processing mode.
!
!    Input matrix is stored in the compressed sparse row (CSR) format with
!    one-based indexing.
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-LOW_ORDER_MOMENTS_CSR_BATCH">
 * \example low_order_moms_csr_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;

/*
 * Input data set parameters
 * Input matrix is stored in the CSR format with one-based indexing
 */
const string datasetFileName = "../data/batch/covcormoments_csr.csv";

void printResults(const low_order_moments::ResultPtr &res);

int main(int argc, char *argv[])
{
    checkArguments(argc, argv, 1, &datasetFileName);

    /* Read datasetFileName from a file and create numeric tables to store the input data */
    CSRNumericTable *dataTable = createSparseTable<float>(datasetFileName);

    /* Create an algorithm to compute low order moments in the batch processing mode using the default method */
    low_order_moments::Batch<float, low_order_moments::fastCSR> algorithm;

    /* Set input objects for the algorithm */
    algorithm.input.set(low_order_moments::data, CSRNumericTablePtr(dataTable));

    /* Compute low order moments */
    algorithm.compute();


    /* Get the computed low order moments */
    low_order_moments::ResultPtr res = algorithm.getResult();

    printResults(res);

    return 0;
}

void printResults(const low_order_moments::ResultPtr &res)
{
    printNumericTable(res->get(low_order_moments::minimum),              "Minimum:");
    printNumericTable(res->get(low_order_moments::maximum),              "Maximum:");
    printNumericTable(res->get(low_order_moments::sum),                  "Sum:");
    printNumericTable(res->get(low_order_moments::sumSquares),           "Sum of squares:");
    printNumericTable(res->get(low_order_moments::sumSquaresCentered),   "Sum of squared difference from the means:");
    printNumericTable(res->get(low_order_moments::mean),                 "Mean:");
    printNumericTable(res->get(low_order_moments::secondOrderRawMoment), "Second order raw moment:");
    printNumericTable(res->get(low_order_moments::variance),             "Variance:");
    printNumericTable(res->get(low_order_moments::standardDeviation),    "Standard deviation:");
    printNumericTable(res->get(low_order_moments::variation),            "Variation:");
}
