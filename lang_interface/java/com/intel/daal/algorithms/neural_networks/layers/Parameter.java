/* file: Parameter.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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

/**
 * @ingroup layers
 * @{
 */
/**
 * @brief Contains classes for the neural network layers
 */
package com.intel.daal.algorithms.neural_networks.layers;

import com.intel.daal.utils.*;
import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;
import com.intel.daal.algorithms.neural_networks.initializers.InitializerIface;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__PARAMETER"></a>
 * @brief Class that specifies parameters of the neural network layer
 */
public class Parameter extends com.intel.daal.algorithms.Parameter {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the parameter of the neural network layer
     * @param context   Context to manage the parameter of the neural network layer
     */
    public Parameter(DaalContext context) {
        super(context);
    }

    public Parameter(DaalContext context, long cParameter) {
        super(context, cParameter);
    }

    /**
     * Sets the layer weights initializer
     * @param weightsInitializer Layer weights initializer
     */
    public void setWeightsInitializer(InitializerIface weightsInitializer) {
        cSetWeightsInitializer(cObject, weightsInitializer.cObject);
    }

    /**
     * Sets the layer biases initializer
     * @param biasesInitializer Layer biases initializer
     */
    public void setBiasesInitializer(InitializerIface biasesInitializer) {
        cSetBiasesInitializer(cObject, biasesInitializer.cObject);
    }

    /**
     *  Gets the flag that specifies whether the weights and biases are initialized or not.
     */
    public boolean getWeightsAndBiasesInitializationFlag() {
        return cGetWeightsAndBiasesInitializationFlag(cObject);
    }

    /**
     *  Sets the flag that specifies whether the weights and biases are initialized or not.
     *  @param weightsAndBiasesInitialized Flag that specifies whether the weights and biases are initialized or not.
     */
    public void setWeightsAndBiasesInitializationFlag(boolean weightsAndBiasesInitialized) {
       cSetWeightsAndBiasesInitializationFlag(cObject, weightsAndBiasesInitialized);
    }

    /**
     *  Gets the flag specifying whether the layer is used for the prediction stage or not
     */
    public boolean getPredictionStage() {
        return cGetPredictionStage(cObject);
    }

    /**
     *  Sets the flag specifying whether the layer is used for the prediction stage or not
     *  @param predictionStage Flag specifying whether the layer is used for the prediction stage or not
     */
    public void setPredictionStage(boolean predictionStage) {
       cSetPredictionStage(cObject, predictionStage);
    }


    private native void cSetWeightsInitializer(long cObject, long cInitializer);
    private native void cSetBiasesInitializer(long cObject, long cInitializer);
    private native boolean cGetWeightsAndBiasesInitializationFlag(long cObject);
    private native void cSetWeightsAndBiasesInitializationFlag(long cObject, boolean weightsAndBiasesInitialized);
    private native boolean cGetPredictionStage(long cObject);
    private native void cSetPredictionStage(long cObject, boolean predictionStage);

}
/** @} */
