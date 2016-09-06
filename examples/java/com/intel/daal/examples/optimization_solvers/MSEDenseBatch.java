/* file: MSEDenseBatch.java */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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
import com.intel.daal.algorithms.optimization_solver.objective_function.ResultCollectionId;
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
        NumericTable data = new HomogenNumericTable(context, Double.class, nFeatures, 0, NumericTable.AllocationFlag.NotAllocate);
        NumericTable dataDependents = new HomogenNumericTable(context, Double.class, 1, 0, NumericTable.AllocationFlag.NotAllocate);
        MergedNumericTable mergedData = new MergedNumericTable(context);
        mergedData.addNumericTable(data);
        mergedData.addNumericTable(dataDependents);

        /* Retrieve the data from an input file */
        dataSource.loadDataBlock(mergedData);

        /* Create an algorithm to compute a MSE */
        Batch algorithm = new Batch(context, Double.class, Method.defaultDense, data.getNumberOfRows());
        algorithm.getInput().set(InputId.data, data);
        algorithm.getInput().set(InputId.dependentVariables, dataDependents);
        algorithm.getInput().set(InputId.argument, new HomogenNumericTable(context, point, nFeatures + 1, 1));
        algorithm.parameter.setResultsToCompute(ResultsToComputeId.gradient | ResultsToComputeId.value | ResultsToComputeId.hessian);

        /* Compute the MSE value and gradient */
        Result result = algorithm.compute();

        Service.printNumericTable("Gradient:", result.get(ResultId.resultCollection, ResultCollectionId.gradientIdx));
        Service.printNumericTable("Value:", result.get(ResultId.resultCollection, ResultCollectionId.valueIdx));
        Service.printNumericTable("Hessian:", result.get(ResultId.resultCollection, ResultCollectionId.hessianIdx));

        context.dispose();
    }
}
