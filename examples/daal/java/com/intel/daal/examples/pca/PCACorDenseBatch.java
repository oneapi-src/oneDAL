/* file: PCACorDenseBatch.java */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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
 //     Java example of principal component analysis (PCA) using the correlation
 //     method in the batch processing mode
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-PCACORRELATIONDENSEBATCH">
 * @example PCACorDenseBatch.java
 */

package com.intel.daal.examples.pca;

import com.intel.daal.algorithms.pca.Batch;
import com.intel.daal.algorithms.pca.InputId;
import com.intel.daal.algorithms.pca.Method;
import com.intel.daal.algorithms.pca.Result;
import com.intel.daal.algorithms.pca.ResultId;
import com.intel.daal.algorithms.pca.ResultsToComputeId;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;


class PCACorDenseBatch {
    /* Input data set parameters */
    private static final String dataset       = "../data/batch/pca_normalized.csv";
    private static DaalContext  context       = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException, java.lang.IllegalArgumentException {
        /* Retrieve the input data from a .csv file */
        FileDataSource dataSource = new FileDataSource(context, dataset,
                                                       DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                                                       DataSource.NumericTableAllocationFlag.DoAllocateNumericTable);
        dataSource.loadDataBlock();

        /* Create an algorithm to compute PCA decomposition using the correlation method */
        Batch pcaAlgorithm = new Batch(context, Float.class, Method.correlationDense);

        /* Set the input data */
        NumericTable data = dataSource.getNumericTable();

        pcaAlgorithm.parameter.setResultsToCompute(ResultsToComputeId.mean | ResultsToComputeId.variance | ResultsToComputeId.eigenvalue);
        pcaAlgorithm.parameter.setIsDeterministic(true);

        /* Set the input data */
        pcaAlgorithm.input.set(InputId.data, data);

        /* Compute PCA decomposition */
        Result res = pcaAlgorithm.compute();

        NumericTable eigenValues = res.get(ResultId.eigenValues);
        NumericTable eigenVectors = res.get(ResultId.eigenVectors);
        NumericTable means = res.get(ResultId.means);
        NumericTable variances = res.get(ResultId.variances);

        Service.printNumericTable("Eigenvalues:", eigenValues);
        Service.printNumericTable("Eigenvectors:", eigenVectors);
        Service.printNumericTable("Means:", means);
        Service.printNumericTable("Variances:", variances);

        context.dispose();
    }
}
