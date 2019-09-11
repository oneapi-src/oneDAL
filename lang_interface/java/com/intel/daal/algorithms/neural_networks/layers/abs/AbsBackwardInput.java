/* file: AbsBackwardInput.java */
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
 * @defgroup abs_layers_backward Backward Absolute Value (Abs) Layer
 * @brief Contains classes of the backward abs layer
 * @ingroup abs_layers
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.abs;

import com.intel.daal.utils.*;
import com.intel.daal.utils.*;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.Tensor;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__ABS__ABSBACKWARDINPUT"></a>
 * @brief Input object for the backward abs layer
 */
public final class AbsBackwardInput extends com.intel.daal.algorithms.neural_networks.layers.BackwardInput {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public AbsBackwardInput(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Sets the input object of the backward abs layer
     * @param id    Identifier of the input object
     * @param val   Value of the input object
     */
    public void set(AbsLayerDataId id, Tensor val) {
        if (id == AbsLayerDataId.auxData) {
            cSetInput(cObject, id.getValue(), val.getCObject());
        }
        else {
            throw new IllegalArgumentException("Incorrect AbsLayerDataId");
        }
    }

    /**
     * Returns the input object of the backward abs layer
     * @param id Identifier of the input object
     * @return   Input object that corresponds to the given identifier
     */
    public Tensor get(AbsLayerDataId id) {
        if (id == AbsLayerDataId.auxData) {
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
