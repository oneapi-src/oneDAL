/* file: LinRegNormEqDenseDistr.java */
/*******************************************************************************
* Copyright 2014-2022 Intel Corporation
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
 //     Java example of multiple linear regression in the distributed processing
 //     mode.
 //
 //     The program trains the multiple linear regression model on a training
 //     data set with the normal equations method and computes regression for
 //     the test data.
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-LINEARREGRESSIONNORMEQDISTRIBUTED">
 * @example LinRegNormEqDenseDistr.java
 */

package com.intel.daal.examples.linear_regression;

import com.intel.daal.algorithms.linear_regression.Model;
import com.intel.daal.algorithms.linear_regression.prediction.*;
import com.intel.daal.algorithms.linear_regression.training.*;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.MergedNumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;

class LinRegNormEqDenseDistr {
    /* Input data set parameters */
    private static final String[] trainDatasetFileNames = {
            "../data/distributed/linear_regression_train_1.csv", "../data/distributed/linear_regression_train_2.csv",
            "../data/distributed/linear_regression_train_3.csv", "../data/distributed/linear_regression_train_4.csv" };

    private static final String testDatasetFileName = "../data/distributed/linear_regression_test.csv";

    private static final int nFeatures            = 10;  /* Number of features in training and testing data sets */
    private static final int nDependentVariables  = 2;   /* Number of dependent variables that correspond to each observation */
    private static final int nNodes               = 4;

    static Model        model;
    static NumericTable results;
    static NumericTable testDependentVariables;

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {

        trainModel();

        testModel();

        printResults();

        context.dispose();
    }

    private static void trainModel() {

        /* Build partial multiple linear regression models on local nodes */
        PartialResult[] pres = new PartialResult[nNodes];

        for (int node = 0; node < nNodes; node++) {
            /* Initialize FileDataSource to retrieve the input data from a .csv file */
            FileDataSource trainDataSource = new FileDataSource(context, trainDatasetFileNames[node],
                    DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                    DataSource.NumericTableAllocationFlag.NotAllocateNumericTable);

            /* Create Numeric Tables for training data and labels */
            NumericTable trainData = new HomogenNumericTable(context, Float.class, nFeatures, 0, NumericTable.AllocationFlag.DoNotAllocate);
            NumericTable trainDependentVariables = new HomogenNumericTable(context, Float.class, nDependentVariables, 0,
                                                                           NumericTable.AllocationFlag.DoNotAllocate);
            MergedNumericTable mergedData = new MergedNumericTable(context);
            mergedData.addNumericTable(trainData);
            mergedData.addNumericTable(trainDependentVariables);

            /* Retrieve the data from an input file */
            trainDataSource.loadDataBlock(mergedData);

            /* Create an algorithm object to train the multiple linear regression model with the normal equations method */
            TrainingDistributedStep1Local linearRegressionTraining = new TrainingDistributedStep1Local(context,
                    Float.class, TrainingMethod.normEqDense);

            /* Set the input data */
            linearRegressionTraining.input.set(TrainingInputId.data, trainData);
            linearRegressionTraining.input.set(TrainingInputId.dependentVariable, trainDependentVariables);

            /* Build a partial multiple linear regression model */
            pres[node] = linearRegressionTraining.compute();
        }

        /* Build the final multiple linear regression model on the master node*/
        /* Create an algorithm object to train the multiple linear regression model with the normal equations method */
        TrainingDistributedStep2Master linearRegressionTraining = new TrainingDistributedStep2Master(context,
                Float.class, TrainingMethod.normEqDense);

        /* Set partial multiple linear regression models built on local nodes */
        for (int node = 0; node < nNodes; node++) {
            linearRegressionTraining.input.add(MasterInputId.partialModels, pres[node]);
        }

        /* Build and retrieve the final multiple linear regression model */
        linearRegressionTraining.compute();

        TrainingResult trainingResult = linearRegressionTraining.finalizeCompute();

        model = trainingResult.get(TrainingResultId.model);
    }

    private static void testModel() {
        /* Initialize FileDataSource to retrieve the input data from a .csv file */
        FileDataSource testDataSource = new FileDataSource(context, testDatasetFileName,
                DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                DataSource.NumericTableAllocationFlag.NotAllocateNumericTable);

        /* Create Numeric Tables for testing data and labels */
        NumericTable testData = new HomogenNumericTable(context, Float.class, nFeatures, 0, NumericTable.AllocationFlag.DoNotAllocate);
        testDependentVariables = new HomogenNumericTable(context, Float.class, nDependentVariables, 0, NumericTable.AllocationFlag.DoNotAllocate);
        MergedNumericTable mergedData = new MergedNumericTable(context);
        mergedData.addNumericTable(testData);
        mergedData.addNumericTable(testDependentVariables);

        /* Retrieve the data from an input file */
        testDataSource.loadDataBlock(mergedData);

        /* Create algorithm objects to predict values of multiple linear regression with the default method */
        PredictionBatch linearRegressionPredict = new PredictionBatch(context, Float.class,
                PredictionMethod.defaultDense);

        /* Provide the input data */
        linearRegressionPredict.input.set(PredictionInputId.data, testData);
        linearRegressionPredict.input.set(PredictionInputId.model, model);

        /* Compute and retrieve the prediction results */
        PredictionResult predictionResult = linearRegressionPredict.compute();

        results = predictionResult.get(PredictionResultId.prediction);
    }

    private static void printResults() {
        NumericTable beta = model.getBeta();
        NumericTable expected = testDependentVariables;
        Service.printNumericTable("Linear Regression coefficients: ", beta);
        Service.printNumericTable("Linear Regression prediction results: (first 10 rows):", results, 10);
        Service.printNumericTable("Ground truth (first 10 rows):", expected, 10);
    }
}
