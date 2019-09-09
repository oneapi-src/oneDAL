/* file: PCAMetricsDenseBatch.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/*
 //  Content:
 //     Java example of PCA quality metrics
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-PCAQUALITYMETRICSETBATCHEXAMPLE">
 * @example PCAMetricsDenseBatch.java
 */

package com.intel.daal.examples.quality_metrics;

import java.nio.DoubleBuffer;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;
import com.intel.daal.algorithms.pca.*;
import com.intel.daal.algorithms.pca.quality_metric.*;
import com.intel.daal.algorithms.pca.quality_metric_set.*;
import com.intel.daal.data_management.data.DataCollection;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;

class PCAMetricsDenseBatch {
    /* Input data set parameters */
    private static final String trainDatasetFileName = "../data/batch/pca_normalized.csv";

    private static final long nVectors = 1000;
    private static final long nComponents = 5;

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {

        /* Initialize FileDataSource to retrieve the input data from a .csv file */
        FileDataSource dataSource = new FileDataSource(context, trainDatasetFileName,
                DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                DataSource.NumericTableAllocationFlag.DoAllocateNumericTable);

        /* Retrieve the data from an input file */
        dataSource.loadDataBlock(nVectors);

        /* Create an algorithm for principal component analysis using the SVD method */
        Batch pca = new Batch(context, Float.class, Method.svdDense);

        /* Set the algorithm input data */
        pca.input.set(InputId.data, dataSource.getNumericTable());

        /* Compute results of the PCA algorithm */
        Result pcaResult = pca.compute();

        /* Create a quality metrics algorithm for explained variances, explained variances ratios and noise_variance */
        QualityMetricSetBatch qms = new QualityMetricSetBatch(context, nComponents, 0);
        ExplainedVarianceInput varianceMetrics = (ExplainedVarianceInput)qms.getInputDataCollection().getInput(QualityMetricId.explainedVariancesMetrics);
        varianceMetrics.set(ExplainedVarianceInputId.eigenValues, pcaResult.get(ResultId.eigenValues));

        /* Compute quality metrics of the PCA algorithm */
        ResultCollection res = qms.compute();

        /* Output quality metrics of the PCA algorithm */
        ExplainedVarianceResult qmsResult = (ExplainedVarianceResult)res.getResult(QualityMetricId.explainedVariancesMetrics);

        Service.printNumericTable("Explained variances:", qmsResult.get(ExplainedVarianceResultId.explainedVariances));
        Service.printNumericTable("Explained variance ratios:", qmsResult.get(ExplainedVarianceResultId.explainedVariancesRatios));
        Service.printNumericTable("Noise variance:", qmsResult.get(ExplainedVarianceResultId.noiseVariance));

        context.dispose();
    }
}
