/* file: SpatialStochasticPooling2dParameter.java */
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

package com.intel.daal.algorithms.neural_networks.layers.spatial_stochastic_pooling2d;

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__SPATIAL_STOCHASTIC_POOLING2D__SPATIALSTOCHASTICPOOLING2DPARAMETER"></a>
 * \brief Class that specifies parameters of the two-dimensional spatial stochastic pooling layer
 */
public class SpatialStochasticPooling2dParameter extends com.intel.daal.algorithms.neural_networks.layers.spatial_pooling2d.SpatialPooling2dParameter {
    /** @private */
    public SpatialStochasticPooling2dParameter(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     *  Returns seed for multinomial distribution random number generator
     */
    public long getSeed() {
        return cGetSeed(cObject);
    }

    /**
     *  Sets the seed for multinomial distribution random number generator
     *  @param seed Seed for multinomial distribution random number generator
     */
    public void setSeed(long seed) {
        cSetSeed(cObject, seed);
    }

    /**
     *  Gets the flag that specifies whether the layer is used for the prediction stage or not
     */
    public boolean getPredictionStage() {
        return cGetPredictionStage(cObject);
    }

    /**
     *  Sets the flag that specifies whether the layer is used for the prediction stage or not
     *  @param predictionStage Flag that specifies whether the layer is used for the prediction stage or not
     */
    public void setPredictionStage(boolean predictionStage) {
        cSetPredictionStage(cObject, predictionStage);
    }

    private native long   cGetSeed(long cParameter);
    private native void   cSetSeed(long cParameter, long seed);
    private native boolean cGetPredictionStage(long cParameter);
    private native void    cSetPredictionStage(long cParameter, boolean predictionStage);
}
