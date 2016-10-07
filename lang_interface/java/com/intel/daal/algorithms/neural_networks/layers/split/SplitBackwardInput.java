/* file: SplitBackwardInput.java */
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

package com.intel.daal.algorithms.neural_networks.layers.split;

import com.intel.daal.data_management.data.HomogenTensor;
import com.intel.daal.data_management.data.Tensor;
import com.intel.daal.data_management.data.KeyValueDataCollection;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__SPLIT__SPLITBACKWARDINPUT"></a>
 * @brief Input object for the backward split layer
 */
public final class SplitBackwardInput extends com.intel.daal.algorithms.neural_networks.layers.BackwardInput {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    public SplitBackwardInput(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Sets the input object of the backward split layer
     * @param id    Identifier of the input object
     * @param val   Value of the input object
     * @param index SplitIndex of the tensor to be set
     */
    public void set(SplitBackwardInputLayerDataId id, Tensor val, long index) {
        if (id == SplitBackwardInputLayerDataId.inputGradientCollection) {
            cSetInput(cObject, id.getValue(), val.getCObject(), index);
        }
        else {
            throw new IllegalArgumentException("Incorrect SplitBackwardInputLayerDataId");
        }
    }

    /**
     * Sets the input object of the backward split layer
     * @param id    Identifier of the input object
     * @param val   Value of the input object
     */
    public void set(SplitBackwardInputLayerDataId id, KeyValueDataCollection val) {
        if (id == SplitBackwardInputLayerDataId.inputGradientCollection) {
            cSetInput(cObject, id.getValue(), val.getCObject());
        }
        else {
            throw new IllegalArgumentException("Incorrect SplitBackwardInputLayerDataId");
        }
    }

    /**
     * Returns the input object of the backward split layer
     * @param id    Identifier of the input object
     * @param index SplitIndex of the tensor to be returned
     * @return      Input object that corresponds to the given identifier
     */
    public Tensor get(SplitBackwardInputLayerDataId id, long index) {
        if (id == SplitBackwardInputLayerDataId.inputGradientCollection) {
            return new HomogenTensor(getContext(), cGetInput(cObject, id.getValue(), index));
        }
        else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
     * Returns the input object of the backward split layer
     * @param id Identifier of the input object
     * @return   Input object that corresponds to the given identifier
     */
    public KeyValueDataCollection get(SplitBackwardInputLayerDataId id) {
        if (id == SplitBackwardInputLayerDataId.inputGradientCollection) {
            return (KeyValueDataCollection)Factory.instance().createObject(getContext(), cGetInput(cObject, id.getValue()));
        }
        else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    private native void cSetInput(long cObject, int id, long ntAddr);
    private native void cSetInput(long cObject, int id, long ntAddr, long index);
    private native long cGetInput(long cObject, int id);
    private native long cGetInput(long cObject, int id, long index);
}
