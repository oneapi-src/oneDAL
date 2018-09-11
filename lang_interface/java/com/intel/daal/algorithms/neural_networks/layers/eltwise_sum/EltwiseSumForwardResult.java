/* file: EltwiseSumForwardResult.java */
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
 * @ingroup eltwise_sum_forward
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
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__ELTWISE_SUM__ELTWISESUMFORWARDRESULT"></a>
 * @brief Class that provides methods to access the result obtained with the compute() method of the forward element-wise sum layer
 */
public final class EltwiseSumForwardResult extends com.intel.daal.algorithms.neural_networks.layers.ForwardResult {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the forward element-wise sum layer result
     * @param context   Context to manage the forward element-wise sum layer result
     */
    public EltwiseSumForwardResult(DaalContext context) {
        super(context);
        this.cObject = cNewResult();
    }

    public EltwiseSumForwardResult(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Returns the result tensor of forward element-wise sum layer
     * @param  id Identifier of the result tensor
     * @return    Result tensor that corresponds to the given identifier
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
     * Returns the result numeric table of the forward element-wise sum layer
     * @param  id Identifier of the result numeric table
     * @return    Result numeric table that corresponds to the given identifier
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
     * Sets the result tensor of forward element-wise sum layer
     * @param id    Identifier of the result tensor
     * @param value Result tensor
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
     * Sets the result numeric table of the forward element-wise sum layer
     * @param id    Identifier of the result numeric table
     * @param value Result numeric tensor
     */
    public void set(EltwiseSumLayerDataNumericTableId id, NumericTable value) {
        if (id == EltwiseSumLayerDataNumericTableId.auxNumberOfCoefficients) {
            cSetNumericTable(cObject, id.getValue(), value.getCObject());
        }
        else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    private native long cNewResult();

    private native long cGetTensor(long cObject, int id);
    private native long cGetNumericTable(long cObject, int id);

    private native void cSetTensor(long cObject, int id, long tensorAddr);
    private native void cSetNumericTable(long cObject, int id, long ntAddr);
}
/** @} */
