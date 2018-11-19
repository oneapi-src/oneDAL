/* file: PCASVDDenseBatch.java */
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
 //  Content:
 //     Java example of principal component analysis (PCA) using the singular
 //     value decomposition (SVD) method in the batch processing mode
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-PCASVDDENSEBATCH">
 * @example PCASVDDenseBatch.java
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


class PCASVDDenseBatch {
    /* Input data set parameters */
    private static final String dataset       = "../data/batch/pca_normalized.csv";

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {
        /* Retrieve the input data from a .csv file */
        FileDataSource dataSource = new FileDataSource(context, dataset,
                                                       DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                                                       DataSource.NumericTableAllocationFlag.DoAllocateNumericTable);
        dataSource.loadDataBlock();

        /* Create an algorithm to compute PCA decomposition using the SVD method */
        Batch pcaAlgorithm = new Batch(context, Float.class, Method.svdDense);

        /* Set the input data */
        NumericTable data = dataSource.getNumericTable();
        pcaAlgorithm.input.set(InputId.data, data);
        pcaAlgorithm.parameter.setResultsToCompute(ResultsToComputeId.mean | ResultsToComputeId.variance | ResultsToComputeId.eigenvalue);
        pcaAlgorithm.parameter.setIsDeterministic(true);

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
