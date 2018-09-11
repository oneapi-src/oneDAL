/* file: KernelFuncRbfDenseBatch.java */
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
 //     Java example of computing a radial basis function (RBF) kernel
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-KERNELFUNCTIONRBFDENSEBATCH">
 * @example KernelFuncRbfDenseBatch.java
 */

package com.intel.daal.examples.kernel_function;

import com.intel.daal.algorithms.kernel_function.InputId;
import com.intel.daal.algorithms.kernel_function.ResultId;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;

class KernelFuncRbfDenseBatch {
    /* Input data set parameters */
    private static final String leftDatasetFileName  = "../data/batch/kernel_function.csv";
    private static final String rightDatasetFileName = "../data/batch/kernel_function.csv";

    private static final double sigma = 1.0;   /* RBF kernel coefficient */

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {
        /* Retrieve the input data */
        FileDataSource leftDataSource = new FileDataSource(context, leftDatasetFileName,
                                                           DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                                                           DataSource.NumericTableAllocationFlag.DoAllocateNumericTable);
        FileDataSource rightDataSource = new FileDataSource(context, rightDatasetFileName,
                                                            DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                                                            DataSource.NumericTableAllocationFlag.DoAllocateNumericTable);
        leftDataSource.loadDataBlock();
        rightDataSource.loadDataBlock();

        /* Create an algorithm */
        com.intel.daal.algorithms.kernel_function.rbf.Batch algorithm = new com.intel.daal.algorithms.kernel_function.rbf.Batch(
            context, Float.class);

        /* Set the kernel algorithm parameter */
        algorithm.parameter.setSigma(sigma);
        algorithm.parameter.setComputationMode(com.intel.daal.algorithms.kernel_function.ComputationMode.matrixMatrix);

        /* Set an input object for the algorithm */
        NumericTable inputX = leftDataSource.getNumericTable();
        NumericTable inputY = rightDataSource.getNumericTable();

        /* Set an input data table for the algorithm */
        algorithm.input.set(InputId.X, inputX);
        algorithm.input.set(InputId.Y, inputY);

        /* Compute the RBF kernel */
        com.intel.daal.algorithms.kernel_function.rbf.Result result = algorithm.compute();

        /* Get the computed results */
        NumericTable values = result.get(ResultId.values);

        /* Print the results */
        Service.printNumericTable("Values", values);

        context.dispose();
    }
}
