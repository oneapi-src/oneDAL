/* file: DfRegHistDenseBatch.java */
/*******************************************************************************
* Copyright 2020 Intel Corporation
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
 //     Java example of decision forest regression.
 //
 //     The program trains the decision forest regression model on a supplied
 //     training data set and then predicts previously unseen data.
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-DfRegHistDenseBatch">
 * @example DfRegHistDenseBatch.java
 */

package com.intel.daal.examples.decision_forest;

import com.intel.daal.algorithms.decision_forest.regression.*;
import com.intel.daal.algorithms.decision_forest.regression.prediction.*;
import com.intel.daal.algorithms.decision_forest.regression.training.*;
import com.intel.daal.algorithms.decision_forest.*;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.MergedNumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;
import com.intel.daal.data_management.data.*;

class DfRegHistDenseBatch {
    /* Input data set parameters */
    private static final String trainDataset = "../data/batch/df_regression_train.csv";

    private static final String testDataset  = "../data/batch/df_regression_test.csv";

    private static final int nFeatures     = 13;

    /* Decision forest regression algorithm parameters */
    private static final int nTrees = 100;

    private static final int maxBins = 256;

    private static final int minBinSize = 5;

    private static NumericTable testGroundTruth;

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {

        TrainingResult trainingResult = trainModel();

        PredictionResult predictionResult = testModel(trainingResult);

        printResults(predictionResult);

        context.dispose();
    }

    private static TrainingResult trainModel() {
        /* Retrieve the data from the input data sets */
        FileDataSource trainDataSource = new FileDataSource(context, trainDataset,
                DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                DataSource.NumericTableAllocationFlag.NotAllocateNumericTable);

        /* Create Numeric Tables for training data and labels */
        NumericTable trainData = new HomogenNumericTable(context, Float.class, nFeatures, 0, NumericTable.AllocationFlag.NotAllocate);
        NumericTable trainGroundTruth = new HomogenNumericTable(context, Float.class, 1, 0, NumericTable.AllocationFlag.NotAllocate);
        MergedNumericTable mergedData = new MergedNumericTable(context);
        mergedData.addNumericTable(trainData);
        mergedData.addNumericTable(trainGroundTruth);

        /* Retrieve the data from an input file */
        trainDataSource.loadDataBlock(mergedData);

        /* Set feature as categorical */
        trainData.getDictionary().setFeature(Float.class,3,DataFeatureUtils.FeatureType.DAAL_CATEGORICAL);

        /* Create algorithm objects to train the decision forest regression model */
        TrainingBatch algorithm = new TrainingBatch(context, Float.class, TrainingMethod.hist);
        algorithm.parameter.setNTrees(nTrees);
        algorithm.parameter.setVariableImportanceMode(VariableImportanceModeId.MDA_Raw);
        algorithm.parameter.setResultsToCompute(ResultsToComputeId.computeOutOfBagError|ResultsToComputeId.computeOutOfBagErrorPerObservation);
        algorithm.parameter.setMaxBins(maxBins);
        algorithm.parameter.setMinBinSize(minBinSize);

        /* Pass a training data set and dependent values to the algorithm */
        algorithm.input.set(InputId.data, trainData);
        algorithm.input.set(InputId.dependentVariable, trainGroundTruth);

        /* Train the decision forest regression model */
        TrainingResult trainingResult = algorithm.compute();

        Service.printNumericTable("Variable importance results: ", trainingResult.get(ResultNumericTableId.variableImportance));
        Service.printNumericTable("OOB error: ", trainingResult.get(ResultNumericTableId.outOfBagError));
        Service.printNumericTable("OOB error per observation (first 10 rows):", trainingResult.get(ResultNumericTableId.outOfBagErrorPerObservation), 10);
        return trainingResult;
    }

    private static PredictionResult testModel(TrainingResult trainingResult) {
        FileDataSource testDataSource = new FileDataSource(context, testDataset,
                DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                DataSource.NumericTableAllocationFlag.NotAllocateNumericTable);

        /* Create Numeric Tables for testing data and labels */
        NumericTable testData = new HomogenNumericTable(context, Float.class, nFeatures, 0, NumericTable.AllocationFlag.NotAllocate);
        testGroundTruth = new HomogenNumericTable(context, Float.class, 1, 0, NumericTable.AllocationFlag.NotAllocate);
        MergedNumericTable mergedData = new MergedNumericTable(context);
        mergedData.addNumericTable(testData);
        mergedData.addNumericTable(testGroundTruth);

        /* Retrieve the data from an input file */
        testDataSource.loadDataBlock(mergedData);
        /* Set feature as categorical */
        testData.getDictionary().setFeature(Float.class,3,DataFeatureUtils.FeatureType.DAAL_CATEGORICAL);

        /* Create algorithm objects for decision forest regression prediction with the fast method */
        PredictionBatch algorithm = new PredictionBatch(context, Float.class, PredictionMethod.defaultDense);

        /* Pass a testing data set and the trained model to the algorithm */
        Model model = trainingResult.get(TrainingResultId.model);
        algorithm.input.set(NumericTableInputId.data, testData);
        algorithm.input.set(ModelInputId.model, model);

        /* Compute prediction results */
        return algorithm.compute();
    }

    private static void printResults(PredictionResult predictionResult) {
        NumericTable predictionResults = predictionResult.get(PredictionResultId.prediction);

        Service.printNumericTable("Decision forest prediction results (first 10 rows):", predictionResults, 10);
        Service.printNumericTable("Ground truth (first 10 rows):", testGroundTruth, 10);
    }

}
