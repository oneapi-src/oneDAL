/* file: out_detect_bacon_dense_batch.cpp */
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
!    C++ example of multivariate outlier detection using the Bacon method
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-OUT_DETECT_BACON_DENSE_BATCH"></a>
 * \example out_detect_bacon_dense_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace algorithms;

typedef double algorithmFPType;     /* Algorithm floating-point type */

/* Input data set parameters */
string datasetFileName = "../data/batch/outlierdetection.csv";

int main(int argc, char *argv[])
{
    checkArguments(argc, argv, 1, &datasetFileName);

    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the test data from a .csv file */
    FileDataSource<CSVFeatureManager> dataSource(datasetFileName, DataSource::doAllocateNumericTable,
                                                 DataSource::doDictionaryFromContext);

    /* Retrieve the data from the input file */
    dataSource.loadDataBlock();

    /* Create an algorithm to detect outliers using the BACON method */
    bacon_outlier_detection::Batch<algorithmFPType, bacon_outlier_detection::defaultDense> algorithm;

    algorithm.input.set(bacon_outlier_detection::data, dataSource.getNumericTable());

    /* Compute outliers */
    algorithm.compute();

    /* Get the computed results */
    bacon_outlier_detection::ResultPtr res = algorithm.getResult();

    printNumericTables(dataSource.getNumericTable().get(), res->get(bacon_outlier_detection::weights).get(),
                       "Input data", "Weights",
                       "Outlier detection result (Bacon method)");

    return 0;
}
