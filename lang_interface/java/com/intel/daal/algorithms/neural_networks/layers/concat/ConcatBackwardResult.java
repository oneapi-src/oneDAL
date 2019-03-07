/* file: ConcatBackwardResult.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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
 * @ingroup concat_backward
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.concat;

import com.intel.daal.utils.*;
import com.intel.daal.data_management.data.Factory;
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
        LibUtils.loadLibrary();
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
            return (Tensor)Factory.instance().createObject(getContext(), cGetValue(cObject, id.getValue(), index));
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
/** @} */
