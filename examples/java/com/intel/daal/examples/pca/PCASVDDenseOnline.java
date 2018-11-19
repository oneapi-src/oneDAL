/* file: PCASVDDenseOnline.java */
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
 //     value decomposition (SVD) method in the online processing mode
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-PCASVDDENSEONLINE">
 * @example PCASVDDenseOnline.java
 */

package com.intel.daal.examples.pca;

import com.intel.daal.algorithms.pca.InputId;
import com.intel.daal.algorithms.pca.Method;
import com.intel.daal.algorithms.pca.Online;
import com.intel.daal.algorithms.pca.Result;
import com.intel.daal.algorithms.pca.ResultId;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;


class PCASVDDenseOnline {
    /* Input data set parameters */
    private static final String dataset         = "../data/online/pca_normalized.csv";
    private static final int    nVectorsInBlock = 250;

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {
        /* Initialize FileDataSource to retrieve the input data from a .csv file */
        FileDataSource dataSource = new FileDataSource(context, dataset,
                                                       DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                                                       DataSource.NumericTableAllocationFlag.DoAllocateNumericTable);

        /* Create an algorithm to compute PCA decomposition using the SVD method */
        Online pcaAlgorithm = new Online(context, Float.class, Method.svdDense);

        /* Set the input data */
        NumericTable data = dataSource.getNumericTable();
        pcaAlgorithm.input.set(InputId.data, data);

        while (dataSource.loadDataBlock(nVectorsInBlock) == nVectorsInBlock) {
            /* Update PCA decomposition */
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
