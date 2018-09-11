/* file: SVDDenseOnline.java */
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
 //     Java example of singular value decomposition (SVD) in the online
 //     processing mode
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-SVDONLINE">
 * @example SVDDenseOnline.java
 */

package com.intel.daal.examples.svd;

import com.intel.daal.algorithms.svd.InputId;
import com.intel.daal.algorithms.svd.Method;
import com.intel.daal.algorithms.svd.Online;
import com.intel.daal.algorithms.svd.Result;
import com.intel.daal.algorithms.svd.ResultId;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;


class SVDDenseOnline {
    /* Input data set parameters */
    private static final String dataset         = "../data/online/svd.csv";
    private static final int    nVectorsInBlock = 4000;

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {
        /* Initialize FileDataSource to retrieve the input data from a .csv file */
        FileDataSource dataSource = new FileDataSource(context, dataset,
                DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                DataSource.NumericTableAllocationFlag.DoAllocateNumericTable);

        /* Create an algorithm to compute SVD in the online processing mode */
        Online svdAlgorithm = new Online(context, Float.class, Method.defaultDense);

        /* Set the input data to the SVD algorithm */
        NumericTable input = dataSource.getNumericTable();
        svdAlgorithm.input.set(InputId.data, input);

        while (dataSource.loadDataBlock(nVectorsInBlock) == nVectorsInBlock) {
            /* Compute SVD */
            svdAlgorithm.compute();
        }

        Result res = svdAlgorithm.finalizeCompute();

        /* Print the results */
        NumericTable leftSingularMatrix = res.get(ResultId.leftSingularMatrix);
        NumericTable singularValues = res.get(ResultId.singularValues);
        NumericTable rightSingularMatrix = res.get(ResultId.rightSingularMatrix);

        Service.printNumericTable("Singular values:", singularValues);
        Service.printNumericTable("Right orthogonal matrix V:", rightSingularMatrix);
        Service.printNumericTable("Left orthogonal matrix U:", leftSingularMatrix, 10);

        context.dispose();
    }
}
