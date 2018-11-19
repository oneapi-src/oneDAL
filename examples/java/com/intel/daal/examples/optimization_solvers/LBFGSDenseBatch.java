/* file: LBFGSDenseBatch.java */
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
 //     Java example of dense LBFGS algorithm in the batch
 //     processing mode
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-LBFGSBATCH">
 * @example LBFGSDenseBatch.java
 */

package com.intel.daal.examples.optimization_solvers;

import com.intel.daal.algorithms.optimization_solver.lbfgs.*;
import com.intel.daal.algorithms.optimization_solver.iterative_solver.InputId;
import com.intel.daal.algorithms.optimization_solver.iterative_solver.Result;
import com.intel.daal.algorithms.optimization_solver.iterative_solver.ResultId;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.MergedNumericTable;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;

class LBFGSDenseBatch {
    private static final long   nFeatures   = 10;
    private static final long   nIterations = 1000;
    private static final double stepLength = 1.0e-4;

    private static double[] initialPoint  = {100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100};
    private static double[] expectedPoint = { 11,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10};

    private static final String datasetFileName = "../data/batch/lbfgs.csv";

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {

        /* Initialize FileDataSource to retrieve the input data from a .csv file */
        FileDataSource dataSource = new FileDataSource(context, datasetFileName,
                DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                DataSource.NumericTableAllocationFlag.NotAllocateNumericTable);

        /* Create Numeric Tables for input data and dependent variables */
        NumericTable data = new HomogenNumericTable(context, Float.class, nFeatures, 0, NumericTable.AllocationFlag.DoNotAllocate);
        NumericTable dependentVariables = new HomogenNumericTable(context, Float.class, 1, 0,
                                                                  NumericTable.AllocationFlag.DoNotAllocate);
        MergedNumericTable mergedData = new MergedNumericTable(context);
        mergedData.addNumericTable(data);
        mergedData.addNumericTable(dependentVariables);

        /* Retrieve the data from input file */
        dataSource.loadDataBlock(mergedData);

        /* Create an MSE objective function for LBFGS */
        com.intel.daal.algorithms.optimization_solver.mse.Batch mseObjectiveFunction =
            new com.intel.daal.algorithms.optimization_solver.mse.Batch(context, Float.class,
                    com.intel.daal.algorithms.optimization_solver.mse.Method.defaultDense, data.getNumberOfRows());

        mseObjectiveFunction.getInput().set(com.intel.daal.algorithms.optimization_solver.mse.InputId.data, data);
        mseObjectiveFunction.getInput().set(com.intel.daal.algorithms.optimization_solver.mse.InputId.dependentVariables,
                                            dependentVariables);

        /* Create objects to compute LBFGS result using the default method */
        Batch algorithm = new Batch(context, Float.class, Method.defaultDense);
        algorithm.parameter.setFunction(mseObjectiveFunction);
        algorithm.parameter.setNIterations(nIterations);
        algorithm.parameter.setStepLengthSequence(
            new HomogenNumericTable(context, Float.class, 1, 1, NumericTable.AllocationFlag.DoAllocate, stepLength));
        algorithm.input.set(InputId.inputArgument, new HomogenNumericTable(context, initialPoint, 1, nFeatures + 1));

        /* Compute LBFGS result */
        Result result = algorithm.compute();

        NumericTable expected = new HomogenNumericTable(context, expectedPoint, 1, nFeatures + 1);
        Service.printNumericTable("Expected coefficients:",          expected);
        Service.printNumericTable("Resulting coefficients:",         result.get(ResultId.minimum));
        Service.printNumericTable("Number of iterations performed:", result.get(ResultId.nIterations));
    }
}
