/* file: LogisticLayerDenseBatch.java */
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
 //     Java example of forward and backward logistic layer usage
 ////////////////////////////////////////////////////////////////////////////////
 */

package com.intel.daal.examples.neural_networks;

import com.intel.daal.algorithms.neural_networks.layers.logistic.*;
import com.intel.daal.algorithms.neural_networks.layers.ForwardResultId;
import com.intel.daal.algorithms.neural_networks.layers.ForwardResultLayerDataId;
import com.intel.daal.algorithms.neural_networks.layers.ForwardInputId;
import com.intel.daal.algorithms.neural_networks.layers.BackwardResultId;
import com.intel.daal.algorithms.neural_networks.layers.ForwardResultId;
import com.intel.daal.algorithms.neural_networks.layers.BackwardInputId;
import com.intel.daal.algorithms.neural_networks.layers.BackwardInputLayerDataId;
import com.intel.daal.data_management.data.Tensor;
import com.intel.daal.data_management.data.HomogenTensor;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-EXAMPLE-JAVA-LOGISTICLAYERBATCH">
 * @example LogisticLayerDenseBatch.java
 */

class LogisticLayerDenseBatch {
    private static final String datasetFileName = "../data/batch/layer.csv";
    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {
        /* Read datasetFileName from a file and create a tensor to store forward input data */
        Tensor tensorData = Service.readTensorFromCSV(context, datasetFileName);

        /* Create an algorithm to compute forward logistic layer results using default method */
        LogisticForwardBatch logisticLayerForward = new LogisticForwardBatch(context, Float.class, LogisticMethod.defaultDense);

        /* Set input objects for the forward logistic layer */
        logisticLayerForward.input.set(ForwardInputId.data, tensorData);

        /* Compute the forward logistic layer results */
        LogisticForwardResult forwardResult = logisticLayerForward.compute();

        /* Print the results of the forward logistic layer */
        Service.printTensor("Forward logistic layer result (first 5 rows):", forwardResult.get(ForwardResultId.value), 5, 0);

        /* Get the size of forward logistic layer output */
        int nSize = (int)forwardResult.get(ForwardResultId.value).getSize();
        long[] dims = forwardResult.get(ForwardResultId.value).getDimensions();

        /* Create a tensor with backward input data */
        double[] data = new double[nSize];
        Tensor tensorDataBack = new HomogenTensor(context, dims, data, 0.01);

        /* Create an algorithm to compute backward logistic layer results using default method */
        LogisticBackwardBatch logisticLayerBackward = new LogisticBackwardBatch(context, Float.class, LogisticMethod.defaultDense);

        /* Set input objects for the backward logistic layer */
        logisticLayerBackward.input.set(BackwardInputId.inputGradient, tensorDataBack);
        logisticLayerBackward.input.set(BackwardInputLayerDataId.inputFromForward, forwardResult.get(ForwardResultLayerDataId.resultForBackward));

        /* Compute backward logistic layer results */
        LogisticBackwardResult backwardResult = logisticLayerBackward.compute();

        /* Print the results of the backward logistic layer */
        Service.printTensor("Backward logistic layer result (first 5 rows):", backwardResult.get(BackwardResultId.gradient), 5, 0);

        context.dispose();
    }
}
