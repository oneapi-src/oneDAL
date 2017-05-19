/* file: ReshapeBackwardInput.java */
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
 * @defgroup reshape_layers_backward Backward Reshape Layer
 * @brief Contains classes of the backward reshape layer
 * @ingroup reshape_layers
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.reshape;

import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.Tensor;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__RESHAPE__RESHAPEBACKWARDINPUT"></a>
 * @brief Input object for the backward reshape layer
 */
public final class ReshapeBackwardInput extends com.intel.daal.algorithms.neural_networks.layers.BackwardInput {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    public ReshapeBackwardInput(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Sets the input object of the backward reshape layer
     * @param id    Identifier of the input object
     * @param val   Value of the input object
     */
    public void set(ReshapeLayerDataId id, Tensor val) {
        if (id == ReshapeLayerDataId.auxData) {
            cSetInput(cObject, id.getValue(), val.getCObject());
        }
        else {
            throw new IllegalArgumentException("Incorrect ReshapeLayerDataId");
        }
    }

    /**
     * Returns the input object of the backward reshape layer
     * @param id Identifier of the input object
     * @return   Input object that corresponds to the given identifier
     */
    public Tensor get(ReshapeLayerDataId id) {
        if (id == ReshapeLayerDataId.auxData) {
            return (Tensor)Factory.instance().createObject(getContext(), cGetInput(cObject, id.getValue()));
        }
        else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    private native void cSetInput(long cObject, int id, long ntAddr);
    private native long cGetInput(long cObject, int id);
}
/** @} */
