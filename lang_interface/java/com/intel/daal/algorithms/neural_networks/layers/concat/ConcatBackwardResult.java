/* file: ConcatBackwardResult.java */
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

import com.intel.daal.data_management.data.HomogenTensor;
import com.intel.daal.data_management.data.Tensor;
import com.intel.daal.algorithms.neural_networks.layers.BackwardResultLayerDataId;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__CONCAT__CONCATBACKWARDRESULT"></a>
 * @brief Provides methods to access results obtained with the compute() method of the backward concat layer
 */
public class ConcatBackwardResult extends com.intel.daal.algorithms.neural_networks.layers.BackwardResult {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs the backward concat layer result
     * @param context   Context to manage the backward concat layer result
     */
    public ConcatBackwardResult(DaalContext context) {
        super(context);
        this.cObject = cNewResult();
    }

    public ConcatBackwardResult(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Returns the result of the backward concat layer
     * @param  id    Identifier of the result
     * @param  index ConcatIndex of the result object
     * @return Result that corresponds to the given identifier
     */
    public Tensor get(BackwardResultLayerDataId id, long index) {
        if (id == BackwardResultLayerDataId.resultLayerData) {
            return new HomogenTensor(getContext(), cGetValue(cObject, id.getValue(), index));
        }
        else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
     * Sets the result of the backward concat layer
     * @param id    Identifier of the result
     * @param val   Result that corresponds to the given identifier
     * @param index ConcatIndex of the result object
     */
    public void set(BackwardResultLayerDataId id, Tensor val, long index) {
        if (id == BackwardResultLayerDataId.resultLayerData) {
            cSetValue(cObject, id.getValue(), val.getCObject(), index);
        }
        else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    private native long cNewResult();

    private native long cGetValue(long cObject, int id, long index);

    private native void cSetValue(long cObject, int id, long ntAddr, long index);
}
