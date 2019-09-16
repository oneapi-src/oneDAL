/* file: StumpRegMseDenseBatch.java */
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
 //     Java example of stump regression.
 //
 //     The program trains the stump model on a supplied training data set and
 //     then performs regression of previously unseen data.
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-STUMP_STUMP_REG_MSE_DENSE_BATCH">
 * @example StumpRegMseDenseBatch.java
 */

package com.intel.daal.examples.stump;

import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;

import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.MergedNumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;

import com.intel.daal.algorithms.regression.training.InputId;
import com.intel.daal.algorithms.regression.training.TrainingResultId;

import com.intel.daal.algorithms.regression.prediction.ModelInputId;
import com.intel.daal.algorithms.regression.prediction.NumericTableInputId;
import com.intel.daal.algorithms.regression.prediction.PredictionResult;
import com.intel.daal.algorithms.regression.prediction.PredictionResultId;

import com.intel.daal.algorithms.stump.regression.Model;
import com.intel.daal.algorithms.stump.regression.training.TrainingBatch;
import com.intel.daal.algorithms.stump.regression.training.TrainingResult;
import com.intel.daal.algorithms.stump.regression.training.TrainingMethod;
import com.intel.daal.algorithms.stump.regression.prediction.PredictionBatch;
import com.intel.daal.algorithms.stump.regression.prediction.PredictionMethod;

class StumpRegMseDenseBatch {

    /* Input data set parameters */
    private static final String trainDatasetFileName = "../data/batch/stump_train.csv";
    private static final String testDatasetFileName  = "../data/batch/stump_test.csv";

    private static final int nFeatures = 20;

    private static TrainingResult   trainingResult;
    private static PredictionResult predictionResult;
    private static NumericTable     testGroundTruth;

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {

        trainModel();

        testModel();

        printResults();

        context.dispose();
    }

    private static void trainModel() {
        /* Retrieve the data from the input data sets */
        FileDataSource trainDataSource = new FileDataSource(context, trainDatasetFileName,
                DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                DataSource.NumericTableAllocationFlag.NotAllocateNumericTable);

        /* Create Numeric Tables for training data and ground truth */
        NumericTable trainData = new HomogenNumericTable(context, Float.class, nFeatures, 0, NumericTable.AllocationFlag.DoNotAllocate);
        NumericTable trainGroundTruth = new HomogenNumericTable(context, Float.class, 1, 0, NumericTable.AllocationFlag.DoNotAllocate);
        MergedNumericTable mergedData = new MergedNumericTable(context);
        mergedData.addNumericTable(trainData);
        mergedData.addNumericTable(trainGroundTruth);

        /* Retrieve the data from an input file */
        trainDataSource.loadDataBlock(mergedData);

        /* Create algorithm objects to train the stump model */
        TrainingBatch algorithm = new TrainingBatch(context, Float.class, TrainingMethod.defaultDense);

        algorithm.input.set(InputId.data, trainData);
        algorithm.input.set(InputId.dependentVariables, trainGroundTruth);

        /* Train the stump model */
        trainingResult = algorithm.compute();
    }

    private static void testModel() {
        FileDataSource testDataSource = new FileDataSource(context, testDatasetFileName,
                DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                DataSource.NumericTableAllocationFlag.NotAllocateNumericTable);

        /* Create Numeric Tables for testing data and ground truth */
        NumericTable testData = new HomogenNumericTable(context, Float.class, nFeatures, 0, NumericTable.AllocationFlag.DoNotAllocate);
        testGroundTruth = new HomogenNumericTable(context, Float.class, 1, 0, NumericTable.AllocationFlag.DoNotAllocate);
        MergedNumericTable mergedData = new MergedNumericTable(context);
        mergedData.addNumericTable(testData);
        mergedData.addNumericTable(testGroundTruth);

        /* Retrieve the data from an input file */
        testDataSource.loadDataBlock(mergedData);

        /* Create algorithm objects to predict values with the fast method */
        PredictionBatch algorithm = new PredictionBatch(context, Float.class, PredictionMethod.defaultDense);

        Model model = trainingResult.get(TrainingResultId.model);

        /* Pass a testing data set and the trained model to the algorithm */
        algorithm.input.set(NumericTableInputId.data, testData);
        algorithm.input.set(ModelInputId.model, model);

        /* Compute the prediction results */
        predictionResult = algorithm.compute();
    }

    private static void printResults() {
        NumericTable predictionResults = predictionResult.get(PredictionResultId.prediction);
        Service.printNumericTables(testGroundTruth, predictionResults,
                                   "Ground truth", "Regression results",
                                   "Stump regression results (first 20 observations):", 20);
    }
}
