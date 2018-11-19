/* file: LcnBackwardInput.java */
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
 * @defgroup lcn_layers_backward Backward Local contrast normalization (LCN) Layer
 * @brief Contains classes for the backward local contrast normalization layer
 * @ingroup lcn_layers
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.lcn;

import com.intel.daal.utils.*;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.Tensor;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LCN__LCNBACKWARDINPUT"></a>
 * @brief Input object for the backward local contrast normalization layer
 */
public final class LcnBackwardInput extends com.intel.daal.algorithms.neural_networks.layers.BackwardInput {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public LcnBackwardInput(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Sets the input object of the backward local contrast normalization layer
     * @param id    Identifier of the input object
     * @param val   Value of the input object
     */
    public void set(LcnLayerDataId id, Tensor val) {
        if (id == LcnLayerDataId.auxCenteredData || id == LcnLayerDataId.auxSigma || id == LcnLayerDataId.auxC || id == LcnLayerDataId.auxInvMax) {
            cSetInput(cObject, id.getValue(), val.getCObject());
        }
        else {
            throw new IllegalArgumentException("Incorrect LcnBackwardInputId");
        }
    }

    /**
     * Returns the input object of the backward local contrast normalization layer
     * @param id Identifier of the input object
     * @return   Input object that corresponds to the given identifier
     */
    public Tensor get(LcnLayerDataId id) {
        if (id == LcnLayerDataId.auxCenteredData || id == LcnLayerDataId.auxSigma || id == LcnLayerDataId.auxC || id == LcnLayerDataId.auxInvMax) {
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
