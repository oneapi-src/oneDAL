/* file: MnNaiveBayesCSRBatch.java */
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
 //     Java example of Naive Bayes classification in the batch processing mode.
 //
 //     The program trains the Naive Bayes model on a supplied training data set
 //     in compressed sparse rows (CSR) format and then performs classification
 //     of previously unseen data.
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-MULTINOMIALNAIVEBAYESCSRBATCH">
 * @example MnNaiveBayesCSRBatch.java
 */

package com.intel.daal.examples.naive_bayes;

import com.intel.daal.algorithms.classifier.prediction.ModelInputId;
import com.intel.daal.algorithms.classifier.prediction.NumericTableInputId;
import com.intel.daal.algorithms.classifier.prediction.PredictionResult;
import com.intel.daal.algorithms.classifier.prediction.PredictionResultId;
import com.intel.daal.algorithms.classifier.training.InputId;
import com.intel.daal.algorithms.classifier.training.TrainingResultId;
import com.intel.daal.algorithms.multinomial_naive_bayes.Model;
import com.intel.daal.algorithms.multinomial_naive_bayes.prediction.*;
import com.intel.daal.algorithms.multinomial_naive_bayes.training.*;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data.CSRNumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;

class MnNaiveBayesCSRBatch {

    /* Input data set parameters */
    private static final String trainDatasetFileName     = "../data/batch/naivebayes_train_csr.csv";
    private static final String trainGroundTruthFileName = "../data/batch/naivebayes_train_labels.csv";

    private static final String testDatasetFileName     = "../data/batch/naivebayes_test_csr.csv";
    private static final String testGroundTruthFileName = "../data/batch/naivebayes_test_labels.csv";

    private static final int  nTrainObservations = 8000;
    private static final int  nTestObservations  = 2000;
    private static final long nClasses           = 20;

    /* Parameters for the Naive Bayes algorithm */
    private static TrainingResult   trainingResult;
    private static PredictionResult predictionResult;

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {

        trainModel();

        testModel();

        printResults();
        context.dispose();
    }

    private static void trainModel() throws java.io.FileNotFoundException, java.io.IOException {
        /* Initialize FileDataSource to retrieve the input data from a .csv file */
        FileDataSource trainGroundTruthSource = new FileDataSource(context, trainGroundTruthFileName,
                DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                DataSource.NumericTableAllocationFlag.DoAllocateNumericTable);

        /* Load the data from the data files */
        CSRNumericTable trainData = Service.createSparseTable(context, trainDatasetFileName);
        trainGroundTruthSource.loadDataBlock();

        /* Create algorithm objects to train the Naive Bayes model */
        TrainingBatch algorithm = new TrainingBatch(context, Float.class, TrainingMethod.fastCSR, nClasses);

        /* Pass a training data set and dependent values to the algorithm */
        algorithm.input.set(InputId.data,   trainData);
        algorithm.input.set(InputId.labels, trainGroundTruthSource.getNumericTable());

        /* Train the Naive Bayes model */
        trainingResult = algorithm.compute();
    }

    private static void testModel() throws java.io.FileNotFoundException, java.io.IOException {

        CSRNumericTable testData = Service.createSparseTable(context, testDatasetFileName);

        /* Create algorithm objects to predict Naive Bayes values with the fastCSR method */
        PredictionBatch algorithm = new PredictionBatch(context, Float.class, PredictionMethod.fastCSR, nClasses);

        /* Pass a testing data set and the trained model to the algorithm */
        algorithm.input.set(NumericTableInputId.data, testData);
        Model model = trainingResult.get(TrainingResultId.model);
        algorithm.input.set(ModelInputId.model, model);

        /* Compute the prediction results */
        predictionResult = algorithm.compute();
    }

    private static void printResults() throws java.io.FileNotFoundException, java.io.IOException {

        FileDataSource testGroundTruth = new FileDataSource(context, testGroundTruthFileName,
                DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                DataSource.NumericTableAllocationFlag.DoAllocateNumericTable);
        testGroundTruth.loadDataBlock();

        NumericTable expected = testGroundTruth.getNumericTable();
        NumericTable prediction = predictionResult.get(PredictionResultId.prediction);
        Service.printClassificationResult(expected, prediction, "Ground truth", "Classification results",
                "NaiveBayes classification results (first 20 observations):", 20);
        System.out.println("");
    }
}
