/* file: ElasticNetDenseBatch.java */
/*******************************************************************************
* Copyright 2014-2021 Intel Corporation
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
 //     Java example of multiple elastic net in the batch processing mode.
 //
 //     The program trains the multiple elastic net model on a training
 //     data set with the normal equations method and computes regression for
 //     the test data.
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-ELASTICNETBATCH">
 * @example ElasticNetDenseBatch.java
 */

package com.intel.daal.examples.elastic_net;

import com.intel.daal.algorithms.elastic_net.Model;
import com.intel.daal.algorithms.elastic_net.prediction.*;
import com.intel.daal.algorithms.elastic_net.training.*;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.MergedNumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;

class ElasticNetDenseBatch {
    /* Input data set parameters */
    private static final String trainDatasetFileName = "../data/batch/linear_regression_train.csv";

    private static final String testDatasetFileName  = "../data/batch/linear_regression_test.csv";

    private static final int nFeatures           = 10;  /* Number of features in training and testing data sets */
    private static final int nDependentVariables = 2;   /* Number of dependent variables that correspond to each observation */

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

        /* Initialize FileDataSource to retrieve the input data from a .csv file */
        FileDataSource trainDataSource = new FileDataSource(context, trainDatasetFileName,
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

        /* Create an algorithm object to train the multiple elastic net model with the normal equations method */
        TrainingBatch elasticNetTrain = new TrainingBatch(context, Float.class, TrainingMethod.defaultDense);

        elasticNetTrain.input.set(TrainingInputId.data, trainData);
        elasticNetTrain.input.set(TrainingInputId.dependentVariable, trainDependentVariables);
        NumericTable penaltyL1 = new HomogenNumericTable(context, Float.class, nDependentVariables, 1, NumericTable.AllocationFlag.DoAllocate, 0.5);
        NumericTable penaltyL2 = new HomogenNumericTable(context, Float.class, nDependentVariables, 1, NumericTable.AllocationFlag.DoAllocate, 0.5);
        elasticNetTrain.parameter.setPenaltyL1(penaltyL1);
        elasticNetTrain.parameter.setPenaltyL2(penaltyL2);

        /* Build the multiple elastic net model */
        TrainingResult trainingResult = elasticNetTrain.compute();

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

        /* Create algorithm objects to predict values of multiple elastic net with the default method */
        PredictionBatch elasticNetPredict = new PredictionBatch(context, Float.class,
                PredictionMethod.defaultDense);

        elasticNetPredict.input.set(PredictionInputId.data, testData);
        elasticNetPredict.input.set(PredictionInputId.model, model);

        /* Compute prediction results */
        PredictionResult predictionResult = elasticNetPredict.compute();

        results = predictionResult.get(PredictionResultId.prediction);
    }

    private static void printResults() {
        NumericTable beta = model.getBeta();
        NumericTable expected = testDependentVariables;
        Service.printNumericTable("Elastic Net coefficients:", beta);
        Service.printNumericTable("Elastic Net prediction results: (first 10 rows):", results, 10);
        Service.printNumericTable("Ground truth (first 10 rows):", expected, 10);
    }
}
