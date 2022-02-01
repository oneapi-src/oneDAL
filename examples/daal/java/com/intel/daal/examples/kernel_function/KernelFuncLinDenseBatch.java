/* file: KernelFuncLinDenseBatch.java */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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
 //     Java example of computing a linear kernel function
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-KERNELFUNCTIONLINEARDENSEBATCH">
 * @example KernelFuncLinDenseBatch.java
 */

package com.intel.daal.examples.kernel_function;

import com.intel.daal.algorithms.kernel_function.InputId;
import com.intel.daal.algorithms.kernel_function.ResultId;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;

class KernelFuncLinDenseBatch {
    /* Input data set parameters */
    private static final String leftDatasetFileName  = "../data/batch/kernel_function.csv";
    private static final String rightDatasetFileName = "../data/batch/kernel_function.csv";

    private static final double k = 1.0;  /* Linear kernel coefficient k */
    private static final double b = 0.0;  /* Linear kernel coefficient b */

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
        com.intel.daal.algorithms.kernel_function.linear.Batch algorithm = new com.intel.daal.algorithms.kernel_function.linear.Batch(
            context, Float.class);

        /* Set an input object for the algorithm */
        NumericTable inputX = leftDataSource.getNumericTable();
        NumericTable inputY = rightDataSource.getNumericTable();

        /* Set the kernel algorithm parameter */
        algorithm.parameter.setK(k);
        algorithm.parameter.setB(b);
        algorithm.parameter.setComputationMode(com.intel.daal.algorithms.kernel_function.ComputationMode.matrixMatrix);

        /* Set an input data table for the algorithm */
        algorithm.input.set(InputId.X, inputX);
        algorithm.input.set(InputId.Y, inputY);

        /* Compute the linear kernel function */
        com.intel.daal.algorithms.kernel_function.linear.Result result = algorithm.compute();

        /* Get the computed results */
        NumericTable values = result.get(ResultId.values);

        /* Print the results */
        Service.printNumericTable("Values", values);

        context.dispose();
    }
}
