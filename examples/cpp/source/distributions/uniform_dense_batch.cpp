/* file: uniform_dense_batch.cpp */
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
!    C++ example of uniform distribution usage
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-UNIFORM_DENSE_BATCH"></a>
 * \example uniform_dense_batch.cpp
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
    uniform::Batch<> uniform;

    /* Set the algorithm input */
    uniform.input.set(distributions::tableToFill, dataTable);

    /* Set the Mersenne Twister engine to the distribution */
    uniform.parameter.engine = mt19937::Batch<>::create(777);

    /* Perform computations */
    uniform.compute();

    /* Print the results */
    printNumericTable(dataTable, "Uniform distribution output:");

    return 0;
}
