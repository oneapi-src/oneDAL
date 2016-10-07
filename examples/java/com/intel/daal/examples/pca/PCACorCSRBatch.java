/* file: PCACorCSRBatch.java */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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
        Batch pcaAlgorithm = new Batch(context, Double.class, Method.correlationDense);

        com.intel.daal.algorithms.covariance.Batch covarianceSparse
            = new com.intel.daal.algorithms.covariance.Batch(context, Double.class, com.intel.daal.algorithms.covariance.Method.fastCSR);
        pcaAlgorithm.parameter.setCovariance(covarianceSparse);

        /* Set the input data */
        pcaAlgorithm.input.set(InputId.data, data);

        /* Compute PCA decomposition */
        Result res = pcaAlgorithm.compute();

        NumericTable eigenValues = res.get(ResultId.eigenValues);
        NumericTable eigenVectors = res.get(ResultId.eigenVectors);
        Service.printNumericTable("Eigenvalues:", eigenValues);
        Service.printNumericTable("Eigenvectors:", eigenVectors);

        context.dispose();
    }
}
