/* file: PCASVDDenseDistr.java */
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
 //     value decomposition (SVD) method in the distributed processing mode
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-PCASVDDENSEDISTRIBUTED">
 * @example PCASVDDenseDistr.java
 */

package com.intel.daal.examples.pca;

import com.intel.daal.algorithms.PartialResult;
import com.intel.daal.algorithms.pca.DistributedStep1Local;
import com.intel.daal.algorithms.pca.DistributedStep2Master;
import com.intel.daal.algorithms.pca.InputId;
import com.intel.daal.algorithms.pca.MasterInputId;
import com.intel.daal.algorithms.pca.Method;
import com.intel.daal.algorithms.pca.Result;
import com.intel.daal.algorithms.pca.ResultId;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;


class PCASVDDenseDistr {
    /* Input data set parameters */
    private static final String[] dataset = { "../data/distributed/pca_normalized_1.csv",
                                              "../data/distributed/pca_normalized_2.csv", "../data/distributed/pca_normalized_3.csv",
                                              "../data/distributed/pca_normalized_4.csv",
                                            };
    private static final int nNodes = 4;

    private static PartialResult[] pres = new PartialResult[nNodes];

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {

        for (int i = 0; i < nNodes; i++) {
            /* Initialize FileDataSource to retrieve the input data from a .csv file */
            FileDataSource dataSource = new FileDataSource(context, dataset[i],
                                                           DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                                                           DataSource.NumericTableAllocationFlag.DoAllocateNumericTable);

            /* Retrieve the data from the input file */
            dataSource.loadDataBlock();

            /* Create an algorithm to compute PCA decomposition using the SVD method on local nodes*/
            DistributedStep1Local pcaLocal = new DistributedStep1Local(context, Float.class, Method.svdDense);

            /* Set the input data on local nodes */
            NumericTable data = dataSource.getNumericTable();
            pcaLocal.input.set(InputId.data, data);

            /* Compute PCA on local nodes */
            pres[i] = pcaLocal.compute();
        }

        /* Create an algorithm to compute PCA decomposition using the SVD method on the master node */
        DistributedStep2Master pcaMaster = new DistributedStep2Master(context, Float.class, Method.svdDense);

        /* Add partial results computed on local nodes to the algorithm on the master node */
        for (int i = 0; i < nNodes; i++) {
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
