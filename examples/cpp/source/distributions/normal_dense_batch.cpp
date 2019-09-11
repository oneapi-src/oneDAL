/* file: normal_dense_batch.cpp */
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
!    C++ example of normal distribution usage
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-NORMAL_DENSE_BATCH"></a>
 * \example normal_dense_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;
using namespace daal::algorithms::engines;
using namespace daal::algorithms::distributions;

int main(int argc, char *argv[])
{
    /* Create input table to fill with random numbers */
    NumericTablePtr dataTable(new HomogenNumericTable<>(1, 10, NumericTable::doAllocate));

    /* Create the algorithm */
    normal::Batch<> normal;

    /* Set the algorithm input */
    normal.input.set(distributions::tableToFill, dataTable);

    /* Set the Mersenne Twister engine to the distribution */
    normal.parameter.engine = mt19937::Batch<>::create(777);

    /* Perform computations */
    normal.compute();

    /* Print the results */
    printNumericTable(dataTable, "Normal distribution output:");

    return 0;
}
