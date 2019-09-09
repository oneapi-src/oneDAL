/* file: SAGALogisticLossDenseBatch.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
 //     Java example of dense SAGA in the batch processing mode
 //     processing mode
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-SAGALOGLOSBATCH">
 * @example SAGALogisticLossDenseBatch.java
 */

package com.intel.daal.examples.optimization_solvers;

import com.intel.daal.algorithms.optimization_solver.saga.*;
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

class SAGALogisticLossDenseBatch {
    private static final int nFeatures = 100;
    private static final double accuracyThreshold = 0.00000001;
    private static final long nIterations = 100000;
    private static double[] initialPoint = new double[nFeatures + 1];
    /* Input data set parameters */
    private static final String dataFileName = "../data/batch/XM_100.csv";
    private static final String groundTruth = "../data/batch/saga_solution_100_features.csv";

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {
        float l1 = 0.06f;
        float l2 = 0.0f;
        boolean intercept = false;
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

        /* Create an Logistic Loss objective function to compute a SAGA */
        com.intel.daal.algorithms.optimization_solver.logistic_loss.Batch logLossFunction =
            new com.intel.daal.algorithms.optimization_solver.logistic_loss.Batch(context, Float.class,
                    com.intel.daal.algorithms.optimization_solver.logistic_loss.Method.defaultDense, data.getNumberOfRows());

        logLossFunction.getInput().set(com.intel.daal.algorithms.optimization_solver.logistic_loss.InputId.data, data);
        logLossFunction.getInput().set(com.intel.daal.algorithms.optimization_solver.logistic_loss.InputId.dependentVariables, dataDependents);
        logLossFunction.parameter.setPenaltyL1(l1);
        logLossFunction.parameter.setPenaltyL2(l2);
        logLossFunction.parameter.setInterceptFlag(intercept);

        /* Create algorithm objects to compute SAGA results */
        Batch sagaAlgorithm = new Batch(context, Float.class, Method.defaultDense);
        sagaAlgorithm.parameter.setFunction(logLossFunction);
        sagaAlgorithm.parameter.setNIterations(nIterations);
        sagaAlgorithm.parameter.setAccuracyThreshold(accuracyThreshold);

        for(int i = 0; i < (nFeatures+1); i++)
           initialPoint[i] = 0;

        sagaAlgorithm.input.set(InputId.inputArgument, new HomogenNumericTable(context, initialPoint, 1, nFeatures + 1));

        /* Compute the SAGA result for Logistic Loss objective function matrix */
        Result result = sagaAlgorithm.compute();

        Service.printNumericTable("Minimum:",  result.get(ResultId.minimum));
        Service.printNumericTable("nIterations:",  result.get(ResultId.nIterations));

        /* Create an Logistic Loss objective function to check SAGA result */
        com.intel.daal.algorithms.optimization_solver.logistic_loss.Batch logLossFunction_check =
           new com.intel.daal.algorithms.optimization_solver.logistic_loss.Batch(context, Float.class,
                   com.intel.daal.algorithms.optimization_solver.logistic_loss.Method.defaultDense, data.getNumberOfRows());

        logLossFunction_check.getInput().set(com.intel.daal.algorithms.optimization_solver.logistic_loss.InputId.data, data);
        logLossFunction_check.getInput().set(com.intel.daal.algorithms.optimization_solver.logistic_loss.InputId.dependentVariables, dataDependents);
        logLossFunction_check.parameter.setPenaltyL1(l1);
        logLossFunction_check.parameter.setPenaltyL2(l2);
        logLossFunction_check.parameter.setInterceptFlag(intercept);
        logLossFunction_check.parameter.setResultsToCompute(com.intel.daal.algorithms.optimization_solver.objective_function.ResultsToComputeId.value);

        FileDataSource groundTruthDS = new FileDataSource(context, groundTruth,
                DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                DataSource.NumericTableAllocationFlag.NotAllocateNumericTable);

        NumericTable groundTruthNT = new HomogenNumericTable(context, Float.class, 1, 0, NumericTable.AllocationFlag.DoNotAllocate);
        groundTruthDS.loadDataBlock(groundTruthNT);

        logLossFunction_check.getInput().set(com.intel.daal.algorithms.optimization_solver.logistic_loss.InputId.argument, groundTruthNT);
        logLossFunction_check.compute();
        Service.printNumericTable("groundTruth:",  logLossFunction_check.getResult().get(com.intel.daal.algorithms.optimization_solver.objective_function.ResultId.valueIdx));

        logLossFunction_check.getInput().set(com.intel.daal.algorithms.optimization_solver.logistic_loss.InputId.argument, result.get(ResultId.minimum));
        logLossFunction_check.compute();
        Service.printNumericTable("value DAAL:",  logLossFunction_check.getResult().get(com.intel.daal.algorithms.optimization_solver.objective_function.ResultId.valueIdx));
        context.dispose();
    }
}
