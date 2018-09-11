/* file: PCACorCSROnline.java */
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
 //     method in the online processing mode for sparse data
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-PCACORRELATIONCSRONLINE">
 * @example PCACorCSROnline.java
 */

package com.intel.daal.examples.pca;

import com.intel.daal.algorithms.pca.InputId;
import com.intel.daal.algorithms.pca.Method;
import com.intel.daal.algorithms.pca.Online;
import com.intel.daal.algorithms.pca.Result;
import com.intel.daal.algorithms.pca.ResultId;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data.CSRNumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;


class PCACorCSROnline {
    /* Input data set parameters */
    private static DaalContext  context = new DaalContext();

    /* Input data set parameters */
    private static final String datasetFileNames[] = new String[] { "../data/online/covcormoments_csr_1.csv",
                                                                    "../data/online/covcormoments_csr_2.csv",
                                                                    "../data/online/covcormoments_csr_3.csv",
                                                                    "../data/online/covcormoments_csr_4.csv"
                                                                  };
    private static final int nBlocks = 4;

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {
        /* Create an algorithm to compute PCA decomposition using the correlation method */
        Online pcaAlgorithm = new Online(context, Float.class, Method.correlationDense);

        com.intel.daal.algorithms.covariance.Online covarianceSparse
            = new com.intel.daal.algorithms.covariance.Online(context, Float.class, com.intel.daal.algorithms.covariance.Method.fastCSR);
        pcaAlgorithm.parameter.setCovariance(covarianceSparse);

        for (int i = 0; i < nBlocks; i++) {
            /* Read the input data from a file */
            CSRNumericTable data = Service.createSparseTable(context, datasetFileNames[i]);

            /* Set the input data */
            pcaAlgorithm.input.set(InputId.data, data);

            /* Compute partial estimates */
            pcaAlgorithm.compute();
        }

        /* Finalize computations and retrieve the results */
        Result res = pcaAlgorithm.finalizeCompute();

        NumericTable eigenValues = res.get(ResultId.eigenValues);
        NumericTable eigenVectors = res.get(ResultId.eigenVectors);
        Service.printNumericTable("Eigenvalues:", eigenValues);
        Service.printNumericTable("Eigenvectors:", eigenVectors);

        context.dispose();
    }
}
