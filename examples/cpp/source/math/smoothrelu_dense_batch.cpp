/* file: smoothrelu_dense_batch.cpp */
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
!    C++ example of SmoothReLU algorithm.
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-SMOOTHRELU_BATCH"></a>
 * \example smoothrelu_dense_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;
using namespace daal::algorithms::math;

/* Input data set parameters */
string datasetName = "../data/batch/covcormoments_dense.csv";

int main()
{
    /* Retrieve the input data */
    FileDataSource<CSVFeatureManager> dataSource(datasetName, DataSource::doAllocateNumericTable, DataSource::doDictionaryFromContext);
    dataSource.loadDataBlock();

    /* Create an algorithm */
    smoothrelu::Batch<> smoothReLU;

    /* Set an input object for the algorithm */
    smoothReLU.input.set(smoothrelu::data, dataSource.getNumericTable());

    /* Compute SmoothReLU function */
    smoothReLU.compute();

    /* Print the results of the algorithm */
    smoothrelu::ResultPtr res = smoothReLU.getResult();
    printNumericTable(res->get(smoothrelu::value), "SmoothReLU result (first 5 rows):", 5);

    return 0;
}
