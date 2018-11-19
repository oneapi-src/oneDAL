/* file: ImplAlsDenseBatch.java */
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
 //     Java example of the implicit alternating least squares (ALS) algorithm in
 //     the batch processing mode.
 //
 //     The program trains the implicit ALS model on a dense training data set.
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-IMPLICITALSDENSEBATCH">
 * @example ImplAlsDenseBatch.java
 */

package com.intel.daal.examples.implicit_als;

import com.intel.daal.algorithms.implicit_als.Model;
import com.intel.daal.algorithms.implicit_als.prediction.ratings.*;
import com.intel.daal.algorithms.implicit_als.training.*;
import com.intel.daal.algorithms.implicit_als.training.init.*;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;

class ImplAlsDenseBatch {
    private static long         nFactors             = 2;
    private static long         nObservations        = 46;
    private static Model        initialModel;
    private static Model        trainedModel;
    private static NumericTable data;
    /* Input data set parameters */
    private static final String trainDatasetFileName = "../data/batch/implicit_als_dense.csv";

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {

        initializeModel();

        trainModel();

        testModel();

        context.dispose();
    }

    private static void initializeModel() {
        /* Read trainDatasetFileName from a file and create a numeric table for storing the input data */
        FileDataSource dataSource = new FileDataSource(context, trainDatasetFileName,
                DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                DataSource.NumericTableAllocationFlag.DoAllocateNumericTable);
        /* Retrieve the input data */
        dataSource.loadDataBlock(nObservations);
        data = dataSource.getNumericTable();

        /* Create an algorithm object to initialize the implicit ALS model with the default method */
        InitBatch initAlgorithm = new InitBatch(context, Float.class, InitMethod.defaultDense);
        initAlgorithm.parameter.setNFactors(nFactors);

        /* Pass a training data set and dependent values to the algorithm */
        initAlgorithm.input.set(InitInputId.data, data);

        /* Initialize the implicit ALS model */
        InitResult initResult = initAlgorithm.compute();

        initialModel = initResult.get(InitResultId.model);
    }

    private static void trainModel() {
        /* Create an algorithm object to train the implicit ALS model with the default method */
        TrainingBatch alsTrain = new TrainingBatch(context, Float.class, TrainingMethod.defaultDense);

        alsTrain.input.set(NumericTableInputId.data, data);
        alsTrain.input.set(ModelInputId.inputModel, initialModel);
        alsTrain.parameter.setNFactors(nFactors);

        /* Build the implicit ALS model */
        TrainingResult trainingResult = alsTrain.compute();

        trainedModel = trainingResult.get(TrainingResultId.model);
    }

    private static void testModel() {
        /* Create an algorithm object to predict recommendations of the implicit ALS model */
        RatingsBatch algorithm = new RatingsBatch(context, Float.class, RatingsMethod.defaultDense);
        algorithm.parameter.setNFactors(nFactors);

        algorithm.input.set(RatingsModelInputId.model, trainedModel);

        RatingsResult result = algorithm.compute();

        NumericTable predictedRatings = result.get(RatingsResultId.prediction);

        Service.printNumericTable("Predicted ratings:", predictedRatings);
    }
}
