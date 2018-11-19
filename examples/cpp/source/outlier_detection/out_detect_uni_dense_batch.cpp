/* file: out_detect_uni_dense_batch.cpp */
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
!    C++ example of univariate outlier detection
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-OUTLIER_DETECTION_UNIVARIATE_BATCH"></a>
 * \example out_detect_uni_dense_batch.cpp
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

    size_t nFeatures = dataSource.getNumberOfColumns();

    univariate_outlier_detection::Batch<> algorithm;

    algorithm.input.set(univariate_outlier_detection::data, dataSource.getNumericTable());

    /* Compute outliers */
    algorithm.compute();

    /* Get the computed results */
    univariate_outlier_detection::ResultPtr res = algorithm.getResult();

    printNumericTable(dataSource.getNumericTable(), "Input data");
    printNumericTable(res->get(univariate_outlier_detection::weights), "Outlier detection result (univariate)");

    return 0;
}
