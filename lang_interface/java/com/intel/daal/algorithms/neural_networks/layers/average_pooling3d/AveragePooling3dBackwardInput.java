/* file: AveragePooling3dBackwardInput.java */
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

package com.intel.daal.algorithms.neural_networks.layers.average_pooling3d;

import com.intel.daal.services.DaalContext;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.NumericTable;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__AVERAGE_POOLING3D__AVERAGEPOOLING3DBACKWARDINPUT"></a>
 * @brief Input object for the backward three-dimensional average pooling layer
 */
public final class AveragePooling3dBackwardInput extends com.intel.daal.algorithms.neural_networks.layers.pooling3d.Pooling3dBackwardInput {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    public AveragePooling3dBackwardInput(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Sets the input object of the backward three-dimensional average pooling layer
     * @param id    Identifier of the input object
     * @param val   Value of the input object
     */
    public void set(AveragePooling3dLayerDataId id, NumericTable val) {
        if (id == AveragePooling3dLayerDataId.auxInputDimensions) {
            cSetInput(cObject, id.getValue(), val.getCObject());
        }
        else {
            throw new IllegalArgumentException("Incorrect AveragePooling3dBackwardInputId");
        }
    }

    /**
     * Returns the input object of the backward three-dimensional average pooling layer
     * @param id Identifier of the input object
     * @return   Input object that corresponds to the given identifier
     */
    public NumericTable get(AveragePooling3dLayerDataId id) {
        if (id == AveragePooling3dLayerDataId.auxInputDimensions) {
            return new HomogenNumericTable(getContext(), cGetInput(cObject, id.getValue()));
        }
        else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    private native void cSetInput(long cObject, int id, long ntAddr);
    private native long cGetInput(long cObject, int id);
}
