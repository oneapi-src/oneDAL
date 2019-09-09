/* file: SVMMultiClassModelBuilder.java */
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
 //     Java example of multi-class support vector machine (SVM) classification model builder
 //
 //     The program builds multi-class support vector machine using model builder and
 //     computes classification for the test data.
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-SVMMULTICLASSMODELBUILDER">
 * @example SVMMultiClassModelBuilder.java
 */

package com.intel.daal.examples.svm;

import java.nio.FloatBuffer;
import com.intel.daal.algorithms.classifier.training.InputId;
import com.intel.daal.algorithms.classifier.prediction.PredictionResult;
import com.intel.daal.algorithms.classifier.prediction.PredictionResultId;
import com.intel.daal.algorithms.classifier.training.TrainingResultId;
import com.intel.daal.algorithms.classifier.prediction.NumericTableInputId;
import com.intel.daal.algorithms.classifier.prediction.ModelInputId;
import com.intel.daal.algorithms.multi_class_classifier.training.TrainingMethod;
import com.intel.daal.algorithms.multi_class_classifier.prediction.*;
import com.intel.daal.algorithms.multi_class_classifier.Model;
import com.intel.daal.algorithms.multi_class_classifier.ModelBuilder;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.MergedNumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;

class SVMMultiClassModelBuilder {
    /* Input data set parameters */
    private static final String [] trainedModelsFileNames = { "../data/batch/svm_multi_class_trained_model_01.csv",
                                                            "../data/batch/svm_multi_class_trained_model_02.csv",
                                                            "../data/batch/svm_multi_class_trained_model_12.csv" };

    private static final String testDatasetFileName  = "../data/batch/multiclass_iris_train.csv";

    private static final long nFeatures            = 4;
    private static final long nClasses             = 3;

    private static final float [] biases = {-0.774F, -1.507F, -7.559F};

    static Model model;
    static NumericTable results;
    static NumericTable testGroundTruth;

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {

        buildModelFromTraining();
        testModel();
        printResults();
        context.dispose();
    }

    public static void buildModelFromTraining() {

        ModelBuilder multiBuilder = new ModelBuilder(context, TrainingMethod.oneAgainstOne, nFeatures, nClasses);

        long imodel = 0;
        for (long iClass = 1; iClass < nClasses; iClass++) {
            for (long jClass = 0; jClass < iClass; jClass++, imodel++) {
                /* Initialize FileDataSource to retrieve the binary classifications models from a .csv file */
                FileDataSource trainDataSource = new FileDataSource(context, trainedModelsFileNames[(int)imodel],
                        DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                        DataSource.NumericTableAllocationFlag.NotAllocateNumericTable);

                /* Create Numeric Tables for supportVectors and classification coefficients */
                NumericTable supportVectors = new HomogenNumericTable(context, Float.class, nFeatures, 0, NumericTable.AllocationFlag.DoNotAllocate);
                NumericTable classificationCoefficients = new HomogenNumericTable(context, Float.class, 1, 0, NumericTable.AllocationFlag.DoNotAllocate);
                MergedNumericTable mergedData = new MergedNumericTable(context);
                mergedData.addNumericTable(supportVectors);
                mergedData.addNumericTable(classificationCoefficients);
                trainDataSource.loadDataBlock(mergedData);

                long nSV = supportVectors.getNumberOfRows();

                com.intel.daal.algorithms.svm.ModelBuilder modelBuilder =
                                    new com.intel.daal.algorithms.svm.ModelBuilder(context, Float.class, nFeatures, nSV);

                /* Write numbers in model */
                FloatBuffer bufferSupportVectors = FloatBuffer.allocate(0);
                bufferSupportVectors = supportVectors.getBlockOfRows(0, nSV, bufferSupportVectors);

                modelBuilder.setSupportVectors(bufferSupportVectors, nSV*nFeatures);

                /* Set classification coefficients */
                FloatBuffer bufferClassCoef = FloatBuffer.allocate(0);
                bufferClassCoef = classificationCoefficients.getBlockOfRows(0, nSV, bufferClassCoef);

                modelBuilder.setClassificationCoefficients(bufferClassCoef, nSV);

                /* Set bias */
                modelBuilder.setBias(biases[(int)imodel]);

                multiBuilder.setTwoClassClassifierModel(jClass, iClass, modelBuilder.getModel());
            }
        }

        model = multiBuilder.getModel();
    }

    private static void testModel() {
        /* Initialize FileDataSource to retrieve the test data from a .csv file */
        FileDataSource testDataSource = new FileDataSource(context, testDatasetFileName,
                DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                DataSource.NumericTableAllocationFlag.NotAllocateNumericTable);

        /* Create Numeric Tables for testing data and ground truth labels */
        NumericTable testData = new HomogenNumericTable(context, Float.class, nFeatures, 0, NumericTable.AllocationFlag.DoNotAllocate);
        testGroundTruth = new HomogenNumericTable(context, Float.class, 1, 0, NumericTable.AllocationFlag.DoNotAllocate);
        MergedNumericTable mergedData = new MergedNumericTable(context);
        mergedData.addNumericTable(testData);
        mergedData.addNumericTable(testGroundTruth);

        /* Load the data from the input file */
        testDataSource.loadDataBlock(mergedData);

        /* Create an algorithm object to predict multi-class SVM values */
        PredictionBatch multiPredict = new PredictionBatch(context, Float.class, PredictionMethod.multiClassClassifierWu, nClasses);

        com.intel.daal.algorithms.svm.prediction.PredictionBatch twoClassPrediction = new com.intel.daal.algorithms.svm.prediction.PredictionBatch(
            context, Float.class, com.intel.daal.algorithms.svm.prediction.PredictionMethod.defaultDense);

        multiPredict.parameter.setPrediction(twoClassPrediction);

        multiPredict.input.set(NumericTableInputId.data, testData);
        multiPredict.input.set(ModelInputId.model, model);

        /* Compute prediction results */
        PredictionResult predictionResult = multiPredict.compute();

        results = predictionResult.get(PredictionResultId.prediction);
    }

    private static void printResults() {
        Service.printClassificationResult(testGroundTruth, results, "Ground truth", "Classification results",
                                "Multi-class SVM classification sample program results (first 20 observations):", 20);
        System.out.println("");
    }
}
