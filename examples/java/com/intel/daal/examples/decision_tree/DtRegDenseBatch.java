/* file: DtRegDenseBatch.java */
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
 //     Java example of decision tree regression
 //
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-DtRegDenseBatch">
 * @example DtRegDenseBatch.java
 */

package com.intel.daal.examples.decision_tree;

import com.intel.daal.algorithms.decision_tree.regression.Model;
import com.intel.daal.algorithms.decision_tree.regression.prediction.*;
import com.intel.daal.algorithms.decision_tree.regression.training.*;
import com.intel.daal.algorithms.decision_tree.*;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.MergedNumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;
import com.intel.daal.data_management.data.*;

class DtRegDenseBatch {
    /* Input data set parameters */
    private static final String trainDataset  ="../data/batch/decision_tree_train.csv";
    private static final String pruneDataset  ="../data/batch/decision_tree_prune.csv";
    private static final String testDataset  = "../data/batch/decision_tree_test.csv";

    private static final int nFeatures     = 5; /* Number of features in training and testing data sets */

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

        /* Create Numeric Tables for training data and dependent variables */
        NumericTable trainData = new HomogenNumericTable(context, Float.class, nFeatures, 0, NumericTable.AllocationFlag.NotAllocate);
        NumericTable trainGroundTruth = new HomogenNumericTable(context, Float.class, 1, 0, NumericTable.AllocationFlag.NotAllocate);
        MergedNumericTable mergedData = new MergedNumericTable(context);
        mergedData.addNumericTable(trainData);
        mergedData.addNumericTable(trainGroundTruth);

        /* Retrieve the data from an input file */
        trainDataSource.loadDataBlock(mergedData);

        /* Retrieve the pruning data from the input data sets */
        FileDataSource pruneDataSource = new FileDataSource(context, pruneDataset,
                DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                DataSource.NumericTableAllocationFlag.NotAllocateNumericTable);

        /* Create Numeric Tables for pruning data and dependent variables */
        NumericTable pruneData = new HomogenNumericTable(context, Float.class, nFeatures, 0, NumericTable.AllocationFlag.NotAllocate);
        NumericTable pruneGroundTruth = new HomogenNumericTable(context, Float.class, 1, 0, NumericTable.AllocationFlag.NotAllocate);
        MergedNumericTable pruneMergedData = new MergedNumericTable(context);
        pruneMergedData.addNumericTable(pruneData);
        pruneMergedData.addNumericTable(pruneGroundTruth);

        /* Retrieve the pruning data from an input file */
        pruneDataSource.loadDataBlock(pruneMergedData);

        /* Create algorithm objects to train the decision tree regression model */
        TrainingBatch algorithm = new TrainingBatch(context, Float.class, TrainingMethod.defaultDense);

        /* Pass the training data set with labels, and pruning dataset with labels to the algorithm */
        algorithm.input.set(TrainingInputId.data, trainData);
        algorithm.input.set(TrainingInputId.dependentVariables, trainGroundTruth);
        algorithm.input.set(TrainingInputId.dataForPruning, pruneData);
        algorithm.input.set(TrainingInputId.dependentVariablesForPruning, pruneGroundTruth);

        /* Train the decision tree regression model */
        TrainingResult trainingResult = algorithm.compute();

        return trainingResult;
    }

    private static PredictionResult testModel(TrainingResult trainingResult) {

        FileDataSource testDataSource = new FileDataSource(context, testDataset,
                DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                DataSource.NumericTableAllocationFlag.NotAllocateNumericTable);

        /* Create Numeric Tables for testing data and dependent variables */
        NumericTable testData = new HomogenNumericTable(context, Float.class, nFeatures, 0, NumericTable.AllocationFlag.NotAllocate);
        testGroundTruth = new HomogenNumericTable(context, Float.class, 1, 0, NumericTable.AllocationFlag.NotAllocate);
        MergedNumericTable mergedData = new MergedNumericTable(context);
        mergedData.addNumericTable(testData);
        mergedData.addNumericTable(testGroundTruth);

        /* Retrieve the data from an input file */
        testDataSource.loadDataBlock(mergedData);

        /* Create algorithm objects for decision tree regression prediction */
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
        Service.printClassificationResult(testGroundTruth, predictionResults, "Ground truth", "Regression results",
                "Decision tree regression results (first 20 observations):", 20);
        System.out.println("");
    }

}
