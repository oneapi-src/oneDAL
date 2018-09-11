/* file: PCACorCSRBatch.java */
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
 //     Java example of principal component analysis (PCA) using the correlation
 //     method in the batch processing mode for sparse data
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-PCACORRELATIONCSRBATCH">
 * @example PCACorCSRBatch.java
 */

package com.intel.daal.examples.pca;

import com.intel.daal.algorithms.pca.Batch;
import com.intel.daal.algorithms.pca.InputId;
import com.intel.daal.algorithms.pca.Method;
import com.intel.daal.algorithms.pca.Result;
import com.intel.daal.algorithms.pca.ResultId;
import com.intel.daal.algorithms.pca.ResultsToComputeId;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data.CSRNumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;

class PCACorCSRBatch {
    /* Input data set parameters */
    private static DaalContext  context = new DaalContext();
    private static final String datasetFileName = "../data/batch/covcormoments_csr.csv";

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {
        /* Read a data set from a file and create a numeric table for storing the input data */
        CSRNumericTable data = Service.createSparseTable(context, datasetFileName);

        /* Create an algorithm to compute PCA decomposition using the correlation method */
        Batch pcaAlgorithm = new Batch(context, Float.class, Method.correlationDense);

        com.intel.daal.algorithms.covariance.Batch covarianceSparse
            = new com.intel.daal.algorithms.covariance.Batch(context, Float.class, com.intel.daal.algorithms.covariance.Method.fastCSR);
        pcaAlgorithm.parameter.setCovariance(covarianceSparse);
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
