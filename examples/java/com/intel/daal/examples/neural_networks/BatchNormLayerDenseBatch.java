/* file: BatchNormLayerDenseBatch.java */
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
 //     Java example of batch normalization layer in the batch processing mode
 ////////////////////////////////////////////////////////////////////////////////
 */

package com.intel.daal.examples.neural_networks;

import com.intel.daal.algorithms.neural_networks.layers.batch_normalization.*;
import com.intel.daal.algorithms.neural_networks.layers.ForwardResultId;
import com.intel.daal.algorithms.neural_networks.layers.ForwardResultLayerDataId;
import com.intel.daal.algorithms.neural_networks.layers.ForwardInputId;
import com.intel.daal.algorithms.neural_networks.layers.BackwardResultId;
import com.intel.daal.algorithms.neural_networks.layers.BackwardInputId;
import com.intel.daal.algorithms.neural_networks.layers.BackwardInputLayerDataId;
import com.intel.daal.data_management.data.Tensor;
import com.intel.daal.data_management.data.HomogenTensor;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-EXAMPLE-JAVA-BATCHNORMALIZATIONLAYERBATCH">
 * @example BatchNormLayerDenseBatch.java
 */
class BatchNormLayerDenseBatch {
    private static final String datasetFileName = "../data/batch/layer.csv";
    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {
        /* Read datasetFileName from a file and create a tensor to store forward input data */
        Tensor data = Service.readTensorFromCSV(context, datasetFileName);

        Service.printTensor("Forward batch normalization layer input (first 5 rows):", data, 5, 0);

        /* Create an algorithm to compute forward batch_normalization layer results using default method */
        BatchNormalizationForwardBatch forwardLayer = new BatchNormalizationForwardBatch(context, Float.class, BatchNormalizationMethod.defaultDense);

        /* Get collection of dimension sizes of the input data tensor */
        long[] dataDims = data.getDimensions();
        int dimensionSize = (int)dataDims[(int)forwardLayer.parameter.getDimension()];

        /* Create a collection of dimension sizes of input weights, biases, population mean and variance tensors */
        long[] dimensionSizes = new long[1];
        dimensionSizes[0] = dimensionSize;

        /* Create input weights, biases, population mean and population variance tensors */
        double[] weightsData = new double[dimensionSize];
        Tensor weights = new HomogenTensor(context, dimensionSizes, weightsData, 1.0);

        double[] biasesData = new double[dimensionSize];
        Tensor biases = new HomogenTensor(context, dimensionSizes, biasesData, 2.0);

        double[] populationMeanData = new double[dimensionSize];
        Tensor populationMean = new HomogenTensor(context, dimensionSizes, populationMeanData, 0.0);

        double[] populationVarianceData = new double[dimensionSize];
        Tensor populationVariance = new HomogenTensor(context, dimensionSizes, populationVarianceData, 0.0);

        /* Set input objects for the forward batch_normalization layer */
        forwardLayer.input.set(ForwardInputId.data, data);
        forwardLayer.input.set(ForwardInputId.weights, weights);
        forwardLayer.input.set(ForwardInputId.biases, biases);
        forwardLayer.input.set(BatchNormalizationForwardInputLayerDataId.populationMean, populationMean);
        forwardLayer.input.set(BatchNormalizationForwardInputLayerDataId.populationVariance, populationVariance);

        /* Compute forward batch_normalization layer results */
        BatchNormalizationForwardResult forwardResult = forwardLayer.compute();

        /* Print the results of the forward batch normalization layer */
        Service.printTensor("Forward batch normalization layer result (first 5 rows):", forwardResult.get(ForwardResultId.value), 5, 0);
        Service.printTensor("Mini-batch mean (first 5 values):", forwardResult.get(BatchNormalizationLayerDataId.auxMean), 5, 0);
        Service.printTensor("Mini-batch standard deviation (first 5 values):", forwardResult.get(BatchNormalizationLayerDataId.auxStandardDeviation), 5, 0);
        Service.printTensor("Population mean (first 5 values):", forwardResult.get(BatchNormalizationLayerDataId.auxPopulationMean), 5, 0);
        Service.printTensor("Population variance (first 5 values):", forwardResult.get(BatchNormalizationLayerDataId.auxPopulationVariance), 5, 0);

        /* Create input gradient tensor for backward batch normalization layer */
        double[] inputGradientData = new double[(int)data.getSize()];
        Tensor inputGradient = new HomogenTensor(context, dataDims, inputGradientData, 10.0);

        /* Create an algorithm to compute backward batch normalization layer results using default method */
        BatchNormalizationBackwardBatch backwardLayer = new BatchNormalizationBackwardBatch(context, Float.class, BatchNormalizationMethod.defaultDense);

        /* Set input objects for the backward batch normalization layer */
        backwardLayer.input.set(BackwardInputId.inputGradient, inputGradient);
        backwardLayer.input.set(BackwardInputLayerDataId.inputFromForward, forwardResult.get(ForwardResultLayerDataId.resultForBackward));

        /* Compute backward batch normalization layer results */
        BatchNormalizationBackwardResult backwardResult = backwardLayer.compute();

        /* Print the results of the backward batch normalization layer */
        Service.printTensor("Backward batch normalization layer result (first 5 rows):", backwardResult.get(BackwardResultId.gradient), 5, 0);
        Service.printTensor("Weight derivatives (first 5 values):", backwardResult.get(BackwardResultId.weightDerivatives), 5, 0);
        Service.printTensor("Bias derivatives (first 5 values):", backwardResult.get(BackwardResultId.biasDerivatives), 5, 0);

        context.dispose();
    }
}
