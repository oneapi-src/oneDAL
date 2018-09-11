/* file: AdagradOptResDenseBatch.java */
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
 //     Java example of dense Adagrad in the batch processing mode
 //     with optional result calculation and its usage in the next run
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-ADAGRAD-OPT-RES-DENSE-BATCH">
 * @example AdagradOptResDenseBatch.java
 */

package com.intel.daal.examples.optimization_solvers;

import com.intel.daal.algorithms.optimization_solver.adagrad.*;
import com.intel.daal.algorithms.optimization_solver.iterative_solver.Input;
import com.intel.daal.algorithms.optimization_solver.iterative_solver.InputId;
import com.intel.daal.algorithms.optimization_solver.iterative_solver.OptionalInputId;
import com.intel.daal.algorithms.optimization_solver.iterative_solver.Result;
import com.intel.daal.algorithms.optimization_solver.iterative_solver.ResultId;
import com.intel.daal.algorithms.optimization_solver.iterative_solver.OptionalResultId;
import com.intel.daal.algorithms.OptionalArgument;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data.MergedNumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;

class AdagradOptResDenseBatch {
    private static final long nFeatures = 3;
    private static final double accuracyThreshold = 0.0000001;
    private static final long nIterations = 1000;
    private static final long batchSize = 1;
    private static final double learningRate = 1;
    private static double[] startPoint = {8, 2, 1, 4};
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

        /* Create an MSE objective function to compute a Adagrad */
        com.intel.daal.algorithms.optimization_solver.mse.Batch mseFunction =
            new com.intel.daal.algorithms.optimization_solver.mse.Batch(context, Float.class,
                    com.intel.daal.algorithms.optimization_solver.mse.Method.defaultDense, data.getNumberOfRows());

        mseFunction.getInput().set(com.intel.daal.algorithms.optimization_solver.mse.InputId.data, data);
        mseFunction.getInput().set(com.intel.daal.algorithms.optimization_solver.mse.InputId.dependentVariables, dataDependents);

        /* Create algorithm objects to compute Adagrad results */
        Batch adagradAlgorithm = new Batch(context, Float.class, Method.defaultDense);
        adagradAlgorithm.parameter.setFunction(mseFunction);
        adagradAlgorithm.parameter.setLearningRate(new HomogenNumericTable(context, Float.class, 1, 1, NumericTable.AllocationFlag.DoAllocate, learningRate));
        adagradAlgorithm.parameter.setNIterations(nIterations/2);
        adagradAlgorithm.parameter.setAccuracyThreshold(accuracyThreshold);
        adagradAlgorithm.parameter.setBatchSize(batchSize);
        adagradAlgorithm.parameter.setOptionalResultRequired(true);
        adagradAlgorithm.input.set(InputId.inputArgument, new HomogenNumericTable(context, startPoint, 1, nFeatures + 1));

        /* Compute the Adagrad result for MSE objective function matrix */
        Result result = adagradAlgorithm.compute();

        Service.printNumericTable("Minimum after first compute():",  result.get(ResultId.minimum));
        Service.printNumericTable("Number of iterations performed:",  result.get(ResultId.nIterations));

        adagradAlgorithm.input.set(InputId.inputArgument, result.get(ResultId.minimum));
        adagradAlgorithm.input.set(OptionalInputId.optionalArgument, result.get(OptionalResultId.optionalResult));

        /* Compute the Adagrad result for MSE objective function matrix */
        result = adagradAlgorithm.compute();

        Service.printNumericTable("Minimum after second compute():",  result.get(ResultId.minimum));
        Service.printNumericTable("Number of iterations performed:",  result.get(ResultId.nIterations));

        context.dispose();
    }
}
