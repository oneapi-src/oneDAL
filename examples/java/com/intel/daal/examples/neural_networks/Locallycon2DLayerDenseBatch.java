/* file: Locallycon2DLayerDenseBatch.java */
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
 //     Java example of 2D locally connected layer in the batch processing mode
 ////////////////////////////////////////////////////////////////////////////////
 */

package com.intel.daal.examples.neural_networks;

import com.intel.daal.algorithms.neural_networks.layers.locallyconnected2d.*;
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
 * <a name="DAAL-EXAMPLE-JAVA-CLOCALLYCON2DLAYERDENSEBATCH">
 * @example Locallycon2DLayerDenseBatch.java
 */
class Locallycon2DLayerDenseBatch {
    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {
        /* Create a collection of dimension sizes of input data */
        long[] dimensionSizes = new long[4];
        dimensionSizes[0] = 2;
        dimensionSizes[1] = 2;
        dimensionSizes[2] = 6;
        dimensionSizes[3] = 8;

        /* Create input daat tensor */
        double[] data = new double[192];
        Tensor dataTensor = new HomogenTensor(context, dimensionSizes, data, 1.0f);

        /* Create an algorithm to compute forward 2D locally connected layer results using default method */
        LocallyConnected2dForwardBatch locallyconnected2DLayerForward = new LocallyConnected2dForwardBatch(context, Float.class, LocallyConnected2dMethod.defaultDense);

        /* Set input objects for the forward 2D locally connected layer */
        locallyconnected2DLayerForward.input.set(ForwardInputId.data, dataTensor);

        /* Compute forward 2D locally connected layer results */
        LocallyConnected2dForwardResult forwardResult = locallyconnected2DLayerForward.compute();

        /* Print the results of the forward 2D locally connected layer */
        Service.printTensor("Forward 2D locally connected layer result (first 5 rows):", forwardResult.get(ForwardResultId.value), 5, 15);
        Service.printTensor("2D locally connected layer weights (first 5 rows):", forwardResult.get(LocallyConnected2dLayerDataId.auxWeights), 5, 15);

        /* Get the size of forward 2D locally connected layer output */
        int nSize = (int)forwardResult.get(ForwardResultId.value).getSize();
        long[] dims = forwardResult.get(ForwardResultId.value).getDimensions();

        /* Create a tensor with backward input data */
        double[] backData = new double[nSize];
        Tensor tensorDataBack = new HomogenTensor(context, dims, backData, 0.01f);

        /* Create an algorithm to compute backward 2D locally connected layer results using default method */
        LocallyConnected2dBackwardBatch locallyconnected2DLayerBackward = new LocallyConnected2dBackwardBatch(context, Float.class, LocallyConnected2dMethod.defaultDense);

        /* Set input objects for the backward 2D locally connected layer */
        locallyconnected2DLayerBackward.input.set(BackwardInputId.inputGradient, tensorDataBack);
        locallyconnected2DLayerBackward.input.set(BackwardInputLayerDataId.inputFromForward,
                                             forwardResult.get(ForwardResultLayerDataId.resultForBackward));

        /* Compute backward 2D locally connected layer results */
        LocallyConnected2dBackwardResult backwardResult = locallyconnected2DLayerBackward.compute();

        /* Print the results of the backward 2D locally connected layer */
        Service.printTensor("2D locally connected layer backpropagation gradient result (first 5 rows):", backwardResult.get(BackwardResultId.gradient), 5, 15);
        Service.printTensor("2D locally connected layer backpropagation weightDerivative result (first 5 rows):", backwardResult.get(BackwardResultId.weightDerivatives), 5, 15);
        Service.printTensor("2D locally connected layer backpropagation biasDerivative result (first 5 rows):", backwardResult.get(BackwardResultId.biasDerivatives), 5, 15);

        context.dispose();
    }
}
