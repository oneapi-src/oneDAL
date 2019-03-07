/* file: pca_metrics_dense_batch.cpp */
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
!    C++ example of pca quality metrics in batch processing mode.
!
!    The program computes PCA and quality
!    metrics for the PCA.
!******************************************************************************/

/**
* <a name="DAAL-EXAMPLE-CPP-PCA_QUALITY_METRIC_SET_BATCH"></a>
* \example pca_metrics_dense_batch.cpp
*/
#include "daal.h"
#include "service.h"
#include <iostream>
using namespace std;
using namespace daal;
using namespace daal::algorithms;
using namespace daal::algorithms::pca::quality_metric;
using namespace daal::algorithms::pca::quality_metric_set;

/* Input data set parameters */
const string dataFileName = "../data/batch/pca_normalized.csv";
const size_t nVectors = 1000;
const size_t nComponents = 5;

int main(int argc, char *argv[])
{
    checkArguments(argc, argv, 1, &dataFileName);

    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> dataSource(dataFileName, DataSource::doAllocateNumericTable,
        DataSource::doDictionaryFromContext);

    /* Retrieve the data from the input file */
    dataSource.loadDataBlock(nVectors);

    /* Create an algorithm for principal component analysis using the SVD method */
    pca::Batch<float, pca::svdDense> algorithm;

    /* Set the algorithm input data */
    algorithm.input.set(pca::data, dataSource.getNumericTable());

    /* Compute results of the PCA algorithm */
    algorithm.compute();

    /* Create a quality metrics algorithm for explained variances, explained variances ratios and noise_variance */
    pca::quality_metric_set::Batch qms(nComponents);

    services::SharedPtr<algorithms::Input> algInput =
        qms.getInputDataCollection()->getInput(explainedVariancesMetrics);

    explained_variance::InputPtr varianceMetrics = explained_variance::Input::cast(algInput);
    varianceMetrics->set(explained_variance::eigenvalues, algorithm.getResult()->get(pca::eigenvalues));

    /* Compute quality metrics of the PCA algorithm */
    qms.compute();

    /* Output quality metrics of the PCA algorithm */
    explained_variance::ResultPtr qmsResult = explained_variance::Result::cast
    (qms.getResultCollection()->getResult(explainedVariancesMetrics));
    printNumericTable(qmsResult->get(explained_variance::explainedVariances),
        "Explained variances:");
    printNumericTable(qmsResult->get(explained_variance::explainedVariancesRatios),
        "Explained variance ratios:");
    printNumericTable(qmsResult->get(explained_variance::noiseVariance),
        "Noise variance:");

    return 0;
}
