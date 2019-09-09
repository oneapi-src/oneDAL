/* file: EltwiseSumBackwardResult.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
 * @ingroup eltwise_sum_backward
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.eltwise_sum;

import com.intel.daal.utils.*;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.Tensor;
import com.intel.daal.algorithms.neural_networks.layers.BackwardResultLayerDataId;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__ELTWISE_SUM__ELTWISESUMBACKWARDRESULT"></a>
 * @brief Provides methods to access results obtained with the compute() method of the backward element-wise sum layer
 */
public class EltwiseSumBackwardResult extends com.intel.daal.algorithms.neural_networks.layers.BackwardResult {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the backward element-wise sum layer result
     * @param context   Context to manage the backward element-wise sum layer result
     */
    public EltwiseSumBackwardResult(DaalContext context) {
        super(context);
        this.cObject = cNewResult();
    }

    public EltwiseSumBackwardResult(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
    * Returns the result tensor of the backward element-wise layer
    * @param  id     Identifier of the result tensor
    * @param  index  Index of the result tensor
    * @return        Input tensor that corresponds to the given identifier
    */
    public Tensor get(BackwardResultLayerDataId id, long index) {
        if (id == BackwardResultLayerDataId.resultLayerData) {
            return (Tensor)Factory.instance().createObject(getContext(), cGetTensor(cObject, id.getValue(), index));
        }
        else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
     * Sets the result tensor for the backward element-wise layer
     * @param id     Identifier of the result tensor
     * @param value  Pointer to the tensor
     * @param index  Index of the result tensor
     */
    public void set(BackwardResultLayerDataId id, Tensor value, long index) {
        if (id == BackwardResultLayerDataId.resultLayerData) {
            cSetTensor(cObject, id.getValue(), value.getCObject(), index);
        }
        else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    private native long cNewResult();

    private native long cGetTensor(long cObject, int id, long index);
    private native void cSetTensor(long cObject, int id, long tensorAddr, long index);
}
/** @} */
