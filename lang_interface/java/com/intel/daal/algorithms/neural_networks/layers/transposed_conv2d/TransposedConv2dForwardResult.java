/* file: TransposedConv2dForwardResult.java */
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
 * @ingroup transposed_conv2d_forward
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.transposed_conv2d;

import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.Tensor;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__TRANSPOSED_CONV2D__TRANSPOSEDCONV2DFORWARDRESULT"></a>
 * @brief Class that provides methods to access the result obtained with the compute() method of the forward 2D transposed convolution layer
 */
public final class TransposedConv2dForwardResult extends com.intel.daal.algorithms.neural_networks.layers.ForwardResult {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs the forward 2D transposed convolution layer result
     * @param context   Context to manage the forward 2D transposed convolution layer result
     */
    public TransposedConv2dForwardResult(DaalContext context) {
        super(context);
        this.cObject = cNewResult();
    }

    /**
    * Constructs the forward 2D transposed convolution layer result
    * @param context   Context to manage the forward 2D transposed convolution layer result
    * @param cObject   Address of C++ forward result
    */
    public TransposedConv2dForwardResult(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Returns the result of the forward 2D transposed convolution layer
     * @param  id   Identifier of the result
     * @return Result that corresponds to the given identifier
     */
    public Tensor get(TransposedConv2dLayerDataId id) {
        if (id == TransposedConv2dLayerDataId.auxData || id == TransposedConv2dLayerDataId.auxWeights) {
            return (Tensor)Factory.instance().createObject(getContext(), cGetValue(cObject, id.getValue()));
        }
        else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
     * Sets the result of the forward 2D transposed convolution layer
     * @param id   Identifier of the result
     * @param val  Result that corresponds to the given identifier
     */
    public void set(TransposedConv2dLayerDataId id, Tensor val) {
        if (id == TransposedConv2dLayerDataId.auxData || id == TransposedConv2dLayerDataId.auxWeights) {
            cSetValue(cObject, id.getValue(), val.getCObject());
        }
        else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    private native long cNewResult();
    private native long cGetValue(long cObject, int id);
    private native void cSetValue(long cObject, int id, long ntAddr);
}
/** @} */
