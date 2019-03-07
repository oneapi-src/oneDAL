/* file: out_detect_mult_dense_batch.cpp */
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
!    C++ example of multivariate outlier detection
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-OUTLIER_DETECTION_MULTIVARIATE_DENSE_BATCH"></a>
 * \example out_detect_mult_dense_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace algorithms;

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

    /* Create an algorithm to detect outliers using the default method */
    multivariate_outlier_detection::Batch<float, multivariate_outlier_detection::defaultDense> algorithm;

    algorithm.input.set(multivariate_outlier_detection::data, dataSource.getNumericTable());

    /* Compute outliers */
    algorithm.compute();

    /* Get the computed results */
    multivariate_outlier_detection::ResultPtr res = algorithm.getResult();

    printNumericTables(dataSource.getNumericTable().get(), res->get(multivariate_outlier_detection::weights).get(),
                       "Input data", "Weights",
                       "Outlier detection result (Default method)");

    return 0;
}
