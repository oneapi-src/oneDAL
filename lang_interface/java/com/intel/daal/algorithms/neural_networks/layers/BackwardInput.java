/* file: BackwardInput.java */
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
 * @ingroup layers_backward
 * @{
 */
/**
 * @brief
 */
package com.intel.daal.algorithms.neural_networks.layers;

import com.intel.daal.utils.*;
import com.intel.daal.utils.*;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.Tensor;
import com.intel.daal.data_management.data.KeyValueDataCollection;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__BACKWARDINPUT"></a>
 * @brief %Input object for the backward layer
 */
public class BackwardInput extends com.intel.daal.algorithms.Input {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public BackwardInput(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Sets the input object of the layer algorithm
     * @param id    Identifier of the input object
     * @param val   Value of the input object
     */
    public void set(BackwardInputId id, Tensor val) {
        if (id == BackwardInputId.inputGradient) {
            cSetInput(cObject, id.getValue(), val.getCObject());
        }
        else {
            throw new IllegalArgumentException("Incorrect BackwardInputId");
        }
    }

    /**
     * Sets the input object of the layer algorithm
     * @param id    Identifier of the input object
     * @param val   Value of the input object
     */
    public void set(BackwardInputLayerDataId id, KeyValueDataCollection val) {
        if (id == BackwardInputLayerDataId.inputFromForward) {
            cSetInput(cObject, id.getValue(), val.getCObject());
        }
        else {
            throw new IllegalArgumentException("Incorrect BackwardInputLayerDataId");
        }
    }

    /**
     * Returns the input object of the layer algorithm
     * @param id Identifier of the input object
     * @return   Input object that corresponds to the given identifier
     */
    public Tensor get(BackwardInputId id) {
        if (id == BackwardInputId.inputGradient) {
            return (Tensor)Factory.instance().createObject(getContext(), cGetInput(cObject, id.getValue()));
        }
        else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
     * Returns the input object of the layer algorithm
     * @param id Identifier of the input object
     * @return   Input object that corresponds to the given identifier
     */
    public KeyValueDataCollection get(BackwardInputLayerDataId id) {
        if (id == BackwardInputLayerDataId.inputFromForward) {
            return (KeyValueDataCollection)Factory.instance().createObject(getContext(), cGetInput(cObject, id.getValue()));
        }
        else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    private native void cSetInput(long cObject, int id, long ntAddr);
    private native long cGetInput(long cObject, int id);
}
/** @} */
