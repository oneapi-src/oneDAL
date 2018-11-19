/* file: CorDenseOnline.java */
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
 //     Java example of dense correlation matrix computation in the online
 //     processing mode
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-CORRELATIONDENSEONLINE">
 * @example CorDenseOnline.java
 */

package com.intel.daal.examples.covariance;

import com.intel.daal.algorithms.covariance.*;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;


class CorDenseOnline {

    /* Input data set parameters */
    private static final String datasetFileName = "../data/online/covcormoments_dense.csv";
    private static final int    nVectorsInBlock = 50;

    private static FileDataSource dataSource;

    private static Result result;

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {
        /* Retrieve the input data from a .csv file */
        dataSource = new FileDataSource(context, datasetFileName,
                DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                DataSource.NumericTableAllocationFlag.DoAllocateNumericTable);

        /* Create algorithm objects to compute a correlation matrix in the online processing mode using the default method */
        Online algorithm = new Online(context, Float.class, Method.defaultDense);

        /* Set input objects for the algorithm */
        NumericTable input = dataSource.getNumericTable();
        algorithm.input.set(InputId.data, input);

        /* Set the parameter to choose the type of the output matrix */
        algorithm.parameter.setOutputMatrixType(OutputMatrixType.correlationMatrix);

        while (dataSource.loadDataBlock(nVectorsInBlock) == nVectorsInBlock) {
            /* Compute partial estimates */
            algorithm.compute();
        }

        /* Finalize the result in the online processing mode */
        result = algorithm.finalizeCompute();

        HomogenNumericTable correlation = (HomogenNumericTable) result.get(ResultId.correlation);
        HomogenNumericTable mean = (HomogenNumericTable) result.get(ResultId.mean);

        Service.printNumericTable("Correlation matrix:", correlation);
        Service.printNumericTable("Mean vector:", mean);

        context.dispose();
    }
}
