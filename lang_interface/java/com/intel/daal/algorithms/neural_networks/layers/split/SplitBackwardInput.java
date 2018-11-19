/* file: SplitBackwardInput.java */
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
 * @defgroup split_backward Backward Split Layer
 * @brief Contains classes for the backward split layer
 * @ingroup split
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.split;

import com.intel.daal.utils.*;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.Tensor;
import com.intel.daal.data_management.data.KeyValueDataCollection;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__SPLIT__SPLITBACKWARDINPUT"></a>
 * @brief Input object for the backward split layer
 */
public final class SplitBackwardInput extends com.intel.daal.algorithms.neural_networks.layers.BackwardInput {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public SplitBackwardInput(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Sets the input object of the backward split layer
     * @param id    Identifier of the input object
     * @param val   Value of the input object
     * @param index SplitIndex of the tensor to be set
     */
    public void set(SplitBackwardInputLayerDataId id, Tensor val, long index) {
        if (id == SplitBackwardInputLayerDataId.inputGradientCollection) {
            cSetInput(cObject, id.getValue(), val.getCObject(), index);
        }
        else {
            throw new IllegalArgumentException("Incorrect SplitBackwardInputLayerDataId");
        }
    }

    /**
     * Sets the input object of the backward split layer
     * @param id    Identifier of the input object
     * @param val   Value of the input object
     */
    public void set(SplitBackwardInputLayerDataId id, KeyValueDataCollection val) {
        if (id == SplitBackwardInputLayerDataId.inputGradientCollection) {
            cSetInput(cObject, id.getValue(), val.getCObject());
        }
        else {
            throw new IllegalArgumentException("Incorrect SplitBackwardInputLayerDataId");
        }
    }

    /**
     * Returns the input object of the backward split layer
     * @param id    Identifier of the input object
     * @param index SplitIndex of the tensor to be returned
     * @return      Input object that corresponds to the given identifier
     */
    public Tensor get(SplitBackwardInputLayerDataId id, long index) {
        if (id == SplitBackwardInputLayerDataId.inputGradientCollection) {
            return (Tensor)Factory.instance().createObject(getContext(), cGetInput(cObject, id.getValue(), index));
        }
        else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
     * Returns the input object of the backward split layer
     * @param id Identifier of the input object
     * @return   Input object that corresponds to the given identifier
     */
    public KeyValueDataCollection get(SplitBackwardInputLayerDataId id) {
        if (id == SplitBackwardInputLayerDataId.inputGradientCollection) {
            return (KeyValueDataCollection)Factory.instance().createObject(getContext(), cGetInput(cObject, id.getValue()));
        }
        else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    private native void cSetInput(long cObject, int id, long ntAddr);
    private native void cSetInput(long cObject, int id, long ntAddr, long index);
    private native long cGetInput(long cObject, int id);
    private native long cGetInput(long cObject, int id, long index);
}
/** @} */
