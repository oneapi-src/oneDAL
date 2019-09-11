/* file: zscore_dense_batch.cpp */
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
!    C++ example of Z-score normalization algorithm.
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-ZSCORE_BATCH"></a>
 * \example zscore_dense_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;
using namespace daal::algorithms::normalization;

/* Input data set parameters */
string datasetName = "../data/batch/normalization.csv";

int main()
{
    /* Retrieve the input data */
    FileDataSource<CSVFeatureManager> dataSource(datasetName, DataSource::doAllocateNumericTable, DataSource::doDictionaryFromContext);
    dataSource.loadDataBlock();

    NumericTablePtr data = dataSource.getNumericTable();

    /* Create an algorithm */
    zscore::Batch<float, zscore::sumDense> algorithm;

    /* Set an input object for the algorithm */
    algorithm.input.set(zscore::data, data);

    /* Compute Z-score normalization function */
    algorithm.compute();

    /* Print the results of stage */
    zscore::ResultPtr res = algorithm.getResult();

    printNumericTable(data, "First 10 rows of the input data:", 10);
    printNumericTable(res->get(zscore::normalizedData), "First 10 rows of the z-score normalization result:", 10);

    return 0;
}
