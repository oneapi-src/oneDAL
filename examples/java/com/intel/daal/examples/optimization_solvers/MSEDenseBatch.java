/* file: MSEDenseBatch.java */
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
 //     Java example of dense MSE in the batch
 //     processing mode
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-MSEBATCH">
 * @example MSEDenseBatch.java
 */

package com.intel.daal.examples.optimization_solvers;

import com.intel.daal.algorithms.optimization_solver.mse.*;
import com.intel.daal.algorithms.optimization_solver.objective_function.Result;
import com.intel.daal.algorithms.optimization_solver.objective_function.ResultId;
import com.intel.daal.algorithms.optimization_solver.objective_function.ResultsToComputeId;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data.MergedNumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;


class MSEDenseBatch {
    private static final long nFeatures = 3;
    private static double[] point = { -1, 0.1, 0.15, -0.5};
    /* Input data set parameters */
    private static final String dataFileName = "../data/batch/mse.csv";

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {

        /* Retrieve the data from input data sets */
        FileDataSource dataSource = new FileDataSource(context, dataFileName,
                DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                DataSource.NumericTableAllocationFlag.NotAllocateNumericTable);

        /* Create Numeric Tables for data and values for dependent variable */
        NumericTable data = new HomogenNumericTable(context, Float.class, nFeatures, 0, NumericTable.AllocationFlag.DoNotAllocate);
        NumericTable dataDependents = new HomogenNumericTable(context, Float.class, 1, 0, NumericTable.AllocationFlag.DoNotAllocate);
        MergedNumericTable mergedData = new MergedNumericTable(context);
        mergedData.addNumericTable(data);
        mergedData.addNumericTable(dataDependents);

        /* Retrieve the data from an input file */
        dataSource.loadDataBlock(mergedData);

        /* Create an algorithm to compute a MSE */
        Batch algorithm = new Batch(context, Float.class, Method.defaultDense, data.getNumberOfRows());
        algorithm.getInput().set(InputId.data, data);
        algorithm.getInput().set(InputId.dependentVariables, dataDependents);
        algorithm.getInput().set(InputId.argument, new HomogenNumericTable(context, point, 1, nFeatures + 1));
        algorithm.parameter.setResultsToCompute(ResultsToComputeId.gradient | ResultsToComputeId.value | ResultsToComputeId.hessian);

        /* Compute the MSE value and gradient */
        Result result = algorithm.compute();

        Service.printNumericTable("Value", result.get(ResultId.valueIdx));
        Service.printNumericTable("Gradient", result.get(ResultId.gradientIdx));
        Service.printNumericTable("Hessian", result.get(ResultId.hessianIdx));

        context.dispose();
    }
}
