/* file: SVMTwoClassThunderDenseBatch.java */
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
 //     Java example of two-class support vector machine (SVM) classification
 //
 //     The program trains the SVM model on a supplied training data set
 //     in dense format and then performs classification of previously unseen
 //     data.
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-SVMTWOCLASSTHUNDERDENSEBATCH">
 * @example SVMTwoClassThunderDenseBatch.java
 */

package com.intel.daal.examples.svm;

import com.intel.daal.algorithms.classifier.prediction.*;
import com.intel.daal.algorithms.classifier.training.InputId;
import com.intel.daal.algorithms.classifier.training.TrainingResultId;
import com.intel.daal.algorithms.svm.Model;
import com.intel.daal.algorithms.svm.prediction.PredictionBatch;
import com.intel.daal.algorithms.svm.prediction.PredictionMethod;
import com.intel.daal.algorithms.svm.training.TrainingBatch;
import com.intel.daal.algorithms.svm.training.TrainingMethod;
import com.intel.daal.algorithms.svm.training.TrainingResult;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.MergedNumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;

class SVMTwoClassThunderDenseBatch {

    /* Input data set parameters */
    private static final String trainDatasetFileName     = "../data/batch/svm_two_class_train_dense.csv";

    private static final String testDatasetFileName     = "../data/batch/svm_two_class_test_dense.csv";

    private static final int nFeatures     = 20;

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
        /* Retrieve the data from input data sets */
        FileDataSource trainDataSource = new FileDataSource(context, trainDatasetFileName,
                DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                DataSource.NumericTableAllocationFlag.NotAllocateNumericTable);

        /* Create Numeric Tables for training data and labels */
        NumericTable trainData = new HomogenNumericTable(context, Float.class, nFeatures, 0, NumericTable.AllocationFlag.DoNotAllocate);
        NumericTable trainGroundTruth = new HomogenNumericTable(context, Float.class, 1, 0, NumericTable.AllocationFlag.DoNotAllocate);
        MergedNumericTable mergedData = new MergedNumericTable(context);
        mergedData.addNumericTable(trainData);
        mergedData.addNumericTable(trainGroundTruth);

        /* Retrieve the data from an input file */
        trainDataSource.loadDataBlock(mergedData);

        /* Create algorithm objects to train the two-class SVM model */
        TrainingBatch algorithm = new TrainingBatch(context, Float.class, TrainingMethod.thunder);

        /* Set parameters for the two-class SVM algorithm */
        algorithm.parameter
                .setKernel(new com.intel.daal.algorithms.kernel_function.linear.Batch(context, Float.class));

        /* Pass a training data set and dependent values to the algorithm */
        algorithm.input.set(InputId.data, trainData);
        algorithm.input.set(InputId.labels, trainGroundTruth);

        /* Train the two-class SVM model */
        trainingResult = algorithm.compute();
    }

    private static void testModel() {
        FileDataSource testDataSource = new FileDataSource(context, testDatasetFileName,
                DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                DataSource.NumericTableAllocationFlag.NotAllocateNumericTable);

        /* Create Numeric Tables for testing data and labels */
        NumericTable testData = new HomogenNumericTable(context, Float.class, nFeatures, 0, NumericTable.AllocationFlag.DoNotAllocate);
        testGroundTruth = new HomogenNumericTable(context, Float.class, 1, 0, NumericTable.AllocationFlag.DoNotAllocate);
        MergedNumericTable mergedData = new MergedNumericTable(context);
        mergedData.addNumericTable(testData);
        mergedData.addNumericTable(testGroundTruth);

        /* Retrieve the data from an input file */
        testDataSource.loadDataBlock(mergedData);

        /* Create algorithm objects to predict two-class SVM values with the defaultDense method */
        PredictionBatch algorithm = new PredictionBatch(context, Float.class, PredictionMethod.defaultDense);

        algorithm.parameter
                .setKernel(new com.intel.daal.algorithms.kernel_function.linear.Batch(context, Float.class));

        Model model = trainingResult.get(TrainingResultId.model);

        /* Pass a testing data set and the trained model to the algorithm */
        algorithm.input.set(NumericTableInputId.data, testData);
        algorithm.input.set(ModelInputId.model, model);

        /* Compute the prediction results */
        predictionResult = algorithm.compute();
    }

    private static void printResults() {
        NumericTable predictionResults = predictionResult.get(PredictionResultId.prediction);
        Service.printClassificationResult(testGroundTruth, predictionResults, "Ground truth", "Classification results",
                "SVM classification results (first 20 observations):", 20);
        System.out.println("");
    }
}
