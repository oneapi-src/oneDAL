/* file: PCACorCSRDistr.java */
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
 //     method in the distributed processing mode for sparse data
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-PCACORRELATIONCSRDISTRIBUTED">
 * @example PCACorCSRDistr.java
 */

package com.intel.daal.examples.pca;

import com.intel.daal.algorithms.PartialResult;
import com.intel.daal.algorithms.pca.DistributedStep1Local;
import com.intel.daal.algorithms.pca.DistributedStep2Master;
import com.intel.daal.algorithms.pca.PartialCorrelationResult;
import com.intel.daal.algorithms.pca.PartialCorrelationResultID;
import com.intel.daal.algorithms.pca.InputId;
import com.intel.daal.algorithms.pca.MasterInputId;
import com.intel.daal.algorithms.pca.Method;
import com.intel.daal.algorithms.pca.Result;
import com.intel.daal.algorithms.pca.ResultId;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data.CSRNumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;


class PCACorCSRDistr {
    /* Input data set parameters */
    private static final String datasetFileNames[] = new String[] { "../data/distributed/covcormoments_csr_1.csv",
                                                                    "../data/distributed/covcormoments_csr_2.csv",
                                                                    "../data/distributed/covcormoments_csr_3.csv",
                                                                    "../data/distributed/covcormoments_csr_4.csv"
                                                                  };
    private static final int nNodes = 4;

    private static PartialResult[] pres = new PartialResult[nNodes];

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {

        for (int i = 0; i < nNodes; i++) {
            DaalContext localContext = new DaalContext();

            /* Read the input data from a file */
            CSRNumericTable data = Service.createSparseTable(localContext, datasetFileNames[i]);

            /* Create an algorithm to compute PCA decomposition using the correlation method on local nodes */
            DistributedStep1Local pcaLocal = new DistributedStep1Local(localContext, Float.class,
                                                                       Method.correlationDense);

            com.intel.daal.algorithms.covariance.DistributedStep1Local covarianceSparse
                = new com.intel.daal.algorithms.covariance.DistributedStep1Local(localContext, Float.class,
                                                                                 com.intel.daal.algorithms.covariance.Method.fastCSR);
            pcaLocal.parameter.setCovariance(covarianceSparse);

            /* Set the input data on local nodes */
            pcaLocal.input.set(InputId.data, data);

            /* Compute PCA decomposition on local nodes */
            pres[i] = pcaLocal.compute();
            pres[i].pack();

            localContext.dispose();
        }

        /* Create an algorithm to compute PCA decomposition using the correlation method on the master node */
        DistributedStep2Master pcaMaster = new DistributedStep2Master(context, Float.class, Method.correlationDense);

        com.intel.daal.algorithms.covariance.DistributedStep2Master covarianceSparse
            = new com.intel.daal.algorithms.covariance.DistributedStep2Master(context, Float.class,
                                                                              com.intel.daal.algorithms.covariance.Method.fastCSR);
        pcaMaster.parameter.setCovariance(covarianceSparse);

        /* Add partial results computed on local nodes to the algorithm on the master node */
        for (int i = 0; i < nNodes; i++) {
            pres[i].unpack(context);
            pcaMaster.input.add(MasterInputId.partialResults, pres[i]);
        }

        /* Compute PCA decomposition on the master node */
        pcaMaster.compute();

        /* Finalize computations and retrieve the results */
        Result res = pcaMaster.finalizeCompute();

        NumericTable eigenValues = res.get(ResultId.eigenValues);
        NumericTable eigenVectors = res.get(ResultId.eigenVectors);
        Service.printNumericTable("Eigenvalues:", eigenValues);
        Service.printNumericTable("Eigenvectors:", eigenVectors);

        context.dispose();
    }
}
