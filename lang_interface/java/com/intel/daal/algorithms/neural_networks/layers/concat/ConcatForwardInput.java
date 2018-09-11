/* file: ConcatForwardInput.java */
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
 * @defgroup concat_forward Forward Concat Layer
 * @brief Contains classes for the forward concat layer
 * @ingroup concat
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.concat;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.neural_networks.layers.ForwardInputLayerDataId;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.Tensor;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__CONCAT__CONCATFORWARDINPUT"></a>
 * @brief %Input object for the forward concat layer
 */
public class ConcatForwardInput extends com.intel.daal.algorithms.neural_networks.layers.ForwardInput {
    /** @private */
    static {
        LibUtils.loadLibrary();
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
            return (Tensor)Factory.instance().createObject(getContext(), cGetInput(cObject, id.getValue(), index));
        }
        else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    private native void cSetInput(long cObject, int id, long ntAddr, long index);
    private native long cGetInput(long cObject, int id, long index);
}
/** @} */
