/* file: Parameter.java */
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

/**
 * @brief Contains classes for the neural network layers
 */
package com.intel.daal.algorithms.neural_networks.layers;

import com.intel.daal.services.DaalContext;
import com.intel.daal.algorithms.neural_networks.initializers.InitializerIface;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__PARAMETER"></a>
 * @brief Class that specifies parameters of the neural network layer
 */
public class Parameter extends com.intel.daal.algorithms.Parameter {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

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
