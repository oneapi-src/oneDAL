/* file: LocallyConnected2dBackwardInput.java */
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
 * @defgroup locallyconnected2d_backward Backward Two-dimensional Locally Connected Layer
 * @brief Contains classes for the backward 2D locally connected layer
 * @ingroup locallyconnected2d
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.locallyconnected2d;

import com.intel.daal.utils.*;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.Tensor;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LOCALLYCONNECTED2D__LOCALLYCONNECTED2DBACKWARDINPUT"></a>
 * @brief Input object for the backward 2D locally connected layer
 */
public final class LocallyConnected2dBackwardInput extends com.intel.daal.algorithms.neural_networks.layers.BackwardInput {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public LocallyConnected2dBackwardInput(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Sets the input object of the backward 2D locally connected layer
     * @param id    Identifier of the input object
     * @param val   Value of the input object
     */
    public void set(LocallyConnected2dLayerDataId id, Tensor val) {
        if (id == LocallyConnected2dLayerDataId.auxData || id == LocallyConnected2dLayerDataId.auxWeights) {
            cSetInput(cObject, id.getValue(), val.getCObject());
        }
        else {
            throw new IllegalArgumentException("Incorrect LocallyConnected2dBackwardInputId");
        }
    }

    /**
     * Returns the input object of the backward 2D locally connected layer
     * @param id Identifier of the input object
     * @return   Input object that corresponds to the given identifier
     */
    public Tensor get(LocallyConnected2dLayerDataId id) {
        if (id == LocallyConnected2dLayerDataId.auxData || id == LocallyConnected2dLayerDataId.auxWeights) {
            return (Tensor)Factory.instance().createObject(getContext(), cGetInput(cObject, id.getValue()));
        }
        else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    private native void cSetInput(long cObject, int id, long ntAddr);
    private native long cGetInput(long cObject, int id);
}
/** @} */
