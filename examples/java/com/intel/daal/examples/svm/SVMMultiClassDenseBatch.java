/* file: SVMMultiClassDenseBatch.java */
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
 //     Java example of multi-class support vector machine (SVM) classification
 //
 //     The program trains multi-class SVM model on a supplied training data set
 //     in dense format and then performs classification of previously unseen
 //     data.
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-SVMMULTICLASSDENSEBATCH">
 * @example SVMMultiClassDenseBatch.java
 */

package com.intel.daal.examples.svm;

import com.intel.daal.algorithms.classifier.prediction.ModelInputId;
import com.intel.daal.algorithms.classifier.prediction.NumericTableInputId;
import com.intel.daal.algorithms.classifier.prediction.PredictionResult;
import com.intel.daal.algorithms.classifier.prediction.PredictionResultId;
import com.intel.daal.algorithms.classifier.training.InputId;
import com.intel.daal.algorithms.classifier.training.TrainingResultId;
import com.intel.daal.algorithms.multi_class_classifier.Model;
import com.intel.daal.algorithms.multi_class_classifier.prediction.*;
import com.intel.daal.algorithms.multi_class_classifier.training.*;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.MergedNumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;

class SVMMultiClassDenseBatch {

    /* Input data set parameters */
    private static final String trainDatasetFileName = "../data/batch/svm_multi_class_train_dense.csv";

    private static final String testDatasetFileName  = "../data/batch/svm_multi_class_test_dense.csv";

    private static final int nFeatures     = 20;
    private static final int nClasses      = 5;

    private static TrainingResult   trainingResult;
    private static PredictionResult predictionResult;
    private static NumericTable     testGroundTruth;

    private static com.intel.daal.algorithms.svm.training.TrainingBatch twoClassTraining;
    private static com.intel.daal.algorithms.svm.prediction.PredictionBatch twoClassPrediction;

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {

        trainModel();

        testModel();

        printResults();

        context.dispose();
    }

    private static void trainModel() {

        twoClassTraining = new com.intel.daal.algorithms.svm.training.TrainingBatch(
                context, Double.class, com.intel.daal.algorithms.svm.training.TrainingMethod.boser);

        twoClassPrediction = new com.intel.daal.algorithms.svm.prediction.PredictionBatch(
                context, Double.class, com.intel.daal.algorithms.svm.prediction.PredictionMethod.defaultDense);

        /* Retrieve the data from input data sets */
        FileDataSource trainDataSource = new FileDataSource(context, trainDatasetFileName,
                DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                DataSource.NumericTableAllocationFlag.NotAllocateNumericTable);

        /* Create Numeric Tables for training data and labels */
        NumericTable trainData = new HomogenNumericTable(context, Double.class, nFeatures, 0, NumericTable.AllocationFlag.NotAllocate);
        NumericTable trainGroundTruth = new HomogenNumericTable(context, Double.class, 1, 0, NumericTable.AllocationFlag.NotAllocate);
        MergedNumericTable mergedData = new MergedNumericTable(context);
        mergedData.addNumericTable(trainData);
        mergedData.addNumericTable(trainGroundTruth);

        /* Retrieve the data from an input file */
        trainDataSource.loadDataBlock(mergedData);

        /* Create an algorithm to train the multi-class SVM model */
        TrainingBatch algorithm = new TrainingBatch(context, Double.class, TrainingMethod.oneAgainstOne);

        /* Set parameters for the multi-class SVM algorithm */
        algorithm.parameter.setNClasses(nClasses);
        algorithm.parameter.setTraining(twoClassTraining);
        algorithm.parameter.setPrediction(twoClassPrediction);

        /* Pass a training data set and dependent values to the algorithm */
        algorithm.input.set(InputId.data, trainData);
        algorithm.input.set(InputId.labels, trainGroundTruth);

        /* Train the multi-class SVM model */
        trainingResult = algorithm.compute();
    }

    private static void testModel() {

        FileDataSource testDataSource = new FileDataSource(context, testDatasetFileName,
                DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                DataSource.NumericTableAllocationFlag.NotAllocateNumericTable);

        /* Create Numeric Tables for testing data and labels */
        NumericTable testData = new HomogenNumericTable(context, Double.class, nFeatures, 0, NumericTable.AllocationFlag.NotAllocate);
        testGroundTruth = new HomogenNumericTable(context, Double.class, 1, 0, NumericTable.AllocationFlag.NotAllocate);
        MergedNumericTable mergedData = new MergedNumericTable(context);
        mergedData.addNumericTable(testData);
        mergedData.addNumericTable(testGroundTruth);

        /* Retrieve the data from an input file */
        testDataSource.loadDataBlock(mergedData);

        /* Create a numeric table to store the prediction results */
        PredictionBatch algorithm = new PredictionBatch(context, Double.class, PredictionMethod.multiClassClassifierWu);

        algorithm.parameter.setNClasses(nClasses);
        algorithm.parameter.setTraining(twoClassTraining);
        algorithm.parameter.setPrediction(twoClassPrediction);

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
                "SVM multiclass classification results (first 20 observations):", 20);
    }
}
