/* file: EltwiseSumBackwardResult.java */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
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
