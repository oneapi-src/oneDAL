/* file: EltwiseSumBackwardInput.java */
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
 * @defgroup eltwise_sum_backward Backward Element-wise sum Layer
 * @brief Contains classes for the backward element-wise sum layer
 * @ingroup eltwise_sum
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.eltwise_sum;

import com.intel.daal.utils.*;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.Tensor;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ELTWISESUMBACKWARDINPUT"></a>
 * @brief Input object for the backward element-wise sum layer
 */
public final class EltwiseSumBackwardInput extends com.intel.daal.algorithms.neural_networks.layers.BackwardInput {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public EltwiseSumBackwardInput(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Returns an input tensor for backward element-twise sum layer
     * @param  id Identifier of the input tensor
     * @return    Input tensor that corresponds to the given identifier
     */
    public Tensor get(EltwiseSumLayerDataId id) {
        if (id == EltwiseSumLayerDataId.auxCoefficients) {
            return (Tensor)Factory.instance().createObject(getContext(), cGetTensor(cObject, id.getValue()));
        }
        else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
     * Returns an input numeric table for backward element-wise sum layer
     * @param  id Identifier of the input numeric table
     * @return    Input numeric table that corresponds to the given identifier
     */
    public NumericTable get(EltwiseSumLayerDataNumericTableId id) {
        if (id == EltwiseSumLayerDataNumericTableId.auxNumberOfCoefficients) {
            return (NumericTable)Factory.instance().createObject(getContext(), cGetNumericTable(cObject, id.getValue()));
        }
        else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
     * Sets an input tensor for the backward element-twise sum layer
     * @param id    Identifier of the input tensor
     * @param value Input tensor to set
     */
    public void set(EltwiseSumLayerDataId id, Tensor value) {
        if (id == EltwiseSumLayerDataId.auxCoefficients) {
            cSetTensor(cObject, id.getValue(), value.getCObject());
        }
        else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
     * Sets an input numeric table for the backward element-wise sum layer
     * @param id    Identifier of the input numeric table
     * @param value Input numeric table
     */
    public void set(EltwiseSumLayerDataNumericTableId id, NumericTable value) {
        if (id == EltwiseSumLayerDataNumericTableId.auxNumberOfCoefficients) {
            cSetNumericTable(cObject, id.getValue(), value.getCObject());
        }
        else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    private native long cGetTensor(long cObject, int id);
    private native long cGetNumericTable(long cObject, int id);

    private native void cSetTensor(long cObject, int id, long tensorAddr);
    private native void cSetNumericTable(long cObject, int id, long ntAddr);
}
/** @} */
