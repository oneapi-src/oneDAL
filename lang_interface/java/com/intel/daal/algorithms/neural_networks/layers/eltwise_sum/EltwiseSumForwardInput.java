/* file: EltwiseSumForwardInput.java */
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
 * @defgroup eltwise_sum_forward Forward Element-wise Sum Layer
 * @brief Contains classes for the forward element-wise sum layer
 * @ingroup eltwise_sum
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.eltwise_sum;

import com.intel.daal.utils.*;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.Tensor;
import com.intel.daal.algorithms.neural_networks.layers.ForwardInputLayerDataId;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__ELTWISE_SUM__ELTWISESUMFORWARDINPUT"></a>
 * @brief %Input object for the forward element-wise sum layer
 */
public class EltwiseSumForwardInput extends com.intel.daal.algorithms.neural_networks.layers.ForwardInput {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public EltwiseSumForwardInput(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
    * Returns an input tensor of the forward element-wise sum layer
    * @param  id    Identifier of the input tensor
    * @param  index Index of the input tensor
    * @return       Input tensor that corresponds to the given identifier
    */
    public Tensor get(ForwardInputLayerDataId id, long index) {
        if (id == ForwardInputLayerDataId.inputLayerData) {
            return (Tensor)Factory.instance().createObject(getContext(), cGetLayerDataTensor(cObject, id.getValue(), index));
        }
        else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
    * Sets an input tensor for the forward element-wise sum layer
    * @param id    Identifier of the input tensor
    * @param value Pointer to the tensor
    * @param index Index of the input tensor
    */
    public void set(ForwardInputLayerDataId id, Tensor value, long index) {
        if (id == ForwardInputLayerDataId.inputLayerData) {
            cSetLayerDataTensor(cObject, id.getValue(), value.getCObject(), index);
        }
        else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
    * Returns an input tensor of the forward element-wise sum layer
    * @param  id Identifier of the input tensor
    * @return    Input tensor that corresponds to the given identifier
    */
    public Tensor get(EltwiseSumForwardInputId id) {
        if (id == EltwiseSumForwardInputId.coefficients) {
            return (Tensor)Factory.instance().createObject(getContext(), cGetEltwiseInputTensor(cObject, id.getValue()));
        }
        else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
    * Sets an input tensor of the forward element-wise sum layer
    * @param id    Identifier of the input tensor
    * @param value Pointer to the tensor
    */
    public void set(EltwiseSumForwardInputId id, Tensor value) {
        if (id == EltwiseSumForwardInputId.coefficients) {
            cSetEltwiseInputTensor(cObject, id.getValue(), value.getCObject());
        }
        else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    private native void cSetLayerDataTensor(long cObject, int id, long tensorAddr, long index);
    private native void cSetEltwiseInputTensor(long cObject, int id, long tensorAddr);

    private native long cGetLayerDataTensor(long cObject, int id, long index);
    private native long cGetEltwiseInputTensor(long cObject, int id);
}
/** @} */
