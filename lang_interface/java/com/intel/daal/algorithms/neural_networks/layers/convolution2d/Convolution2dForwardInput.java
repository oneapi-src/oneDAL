/* file: Convolution2dForwardInput.java */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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
 * @defgroup convolution2d_forward Forward Two-dimensional Convolution Layer
 * @brief Contains classes for the forward 2D convolution layer
 * @ingroup convolution2d
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.convolution2d;

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__CONVOLUTION2D__CONVOLUTION2DFORWARDINPUT"></a>
 * @brief %Input object for the forward 2D convolution layer
 */
public class Convolution2dForwardInput extends com.intel.daal.algorithms.neural_networks.layers.ForwardInput {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs the forward 2D convolution layer input
     * @param context   Context to manage the forward 2D convolution layer input
     * @param cObject   Address of C++ forward input
     */
    public Convolution2dForwardInput(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Returns dimensions of weights tensor
     * @param parameter  Layer parameter
     * @return Dimensions of weights tensor
     */
    public long[] getWeightsSizes(Convolution2dParameter parameter)
    {
        return cGetWeightsSizes(cObject, parameter.getCObject());
    }

    /**
     * Returns dimensions of biases tensor
     * @param parameter  Layer parameter
     * @return Dimensions of biases tensor
     */
    public long[] getBiasesSizes(Convolution2dParameter parameter)
    {
        return cGetBiasesSizes(cObject, parameter.getCObject());
    }

    private native long[] cGetWeightsSizes(long cObject, long cParameter);
    private native long[] cGetBiasesSizes(long cObject, long cParameter);
}
/** @} */
