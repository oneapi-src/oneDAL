/* file: SGDMomentDenseBatch.java */
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
 //     Java example of dense momentum SGD in the batch processing mode
 //     processing mode
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-SGDMOMENTDENSEBATCH">
 * @example SGDMomentDenseBatch.java
 */

package com.intel.daal.examples.optimization_solvers;

import com.intel.daal.algorithms.optimization_solver.sgd.*;
import com.intel.daal.algorithms.optimization_solver.iterative_solver.InputId;
import com.intel.daal.algorithms.optimization_solver.iterative_solver.Result;
import com.intel.daal.algorithms.optimization_solver.iterative_solver.ResultId;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data.MergedNumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;

class SGDMomentDenseBatch {
    private static final long nFeatures = 3;
    private static final double accuracyThreshold = 0.0000001;
    private static final long nIterations = 1000;
    private static final long batchSize = 4;
    private static final double learningRate = 0.5;
    private static double[] initialPoint = {8, 2, 1, 4};

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

        /* Create an MSE objective function to compute a momentum SGD */
        com.intel.daal.algorithms.optimization_solver.mse.Batch mseFunction =
            new com.intel.daal.algorithms.optimization_solver.mse.Batch(context, Float.class,
                    com.intel.daal.algorithms.optimization_solver.mse.Method.defaultDense, data.getNumberOfRows());

        mseFunction.getInput().set(com.intel.daal.algorithms.optimization_solver.mse.InputId.data, data);
        mseFunction.getInput().set(com.intel.daal.algorithms.optimization_solver.mse.InputId.dependentVariables, dataDependents);

        /* Create algorithm objects to compute momentum SGD results */
        Batch sgdAlgorithm = new Batch(context, Float.class, Method.momentum);
        ParameterMomentum algorithmParameter = (ParameterMomentum)sgdAlgorithm.parameter;
        algorithmParameter.setFunction(mseFunction);
        algorithmParameter.setLearningRateSequence(
            new HomogenNumericTable(context, Float.class, 1, 1, NumericTable.AllocationFlag.DoAllocate, learningRate));
        algorithmParameter.setNIterations(nIterations);
        algorithmParameter.setAccuracyThreshold(accuracyThreshold);
        algorithmParameter.setBatchSize(batchSize);
        sgdAlgorithm.input.set(InputId.inputArgument, new HomogenNumericTable(context, initialPoint, 1, nFeatures + 1));

        /* Compute the momentum SGD result for MSE objective function matrix */
        Result result = sgdAlgorithm.compute();

        Service.printNumericTable("Minimum:",  result.get(ResultId.minimum));
        Service.printNumericTable("Number of iterations performed:",  result.get(ResultId.nIterations));

        context.dispose();
    }
}
