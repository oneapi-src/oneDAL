/* file: ConcatForwardInput.java */
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

package com.intel.daal.algorithms.neural_networks.layers.concat;

import com.intel.daal.algorithms.neural_networks.layers.ForwardInputLayerDataId;
import com.intel.daal.data_management.data.HomogenTensor;
import com.intel.daal.data_management.data.Tensor;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__CONCAT__CONCATFORWARDINPUT"></a>
 * @brief %Input object for the forward concat layer
 */
public class ConcatForwardInput extends com.intel.daal.algorithms.neural_networks.layers.ForwardInput {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    public ConcatForwardInput(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Sets the input object of the forward concat layer
     * @param id    Identifier of the input object
     * @param val   Value of the input object
     * @param index ConcatIndex of the input object
     */
    public void set(ForwardInputLayerDataId id, Tensor val, long index) {
        if (id == ForwardInputLayerDataId.inputLayerData) {
            cSetInput(cObject, id.getValue(), val.getCObject(), index);
        }
        else {
            throw new IllegalArgumentException("Incorrect ConcatForwardInputLayerDataId");
        }
    }

    /**
     * Returns the input object of the forward concat layer
     * @param id    Identifier of the input object
     * @param index ConcatIndex of the input object
     * @return Input object that corresponds to the given identifier
     */
    public Tensor get(ForwardInputLayerDataId id, long index) {
        if (id == ForwardInputLayerDataId.inputLayerData) {
            return new HomogenTensor(getContext(), cGetInput(cObject, id.getValue(), index));
        }
        else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    private native void cSetInput(long cObject, int id, long ntAddr, long index);
    private native long cGetInput(long cObject, int id, long index);
}
