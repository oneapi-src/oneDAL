/* file: SmoothreluBackwardInput.java */
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
 * @defgroup smoothrelu_layers_backward Backward Smooth Rectifier Linear Unit (SmoothReLU) Layer
 * @brief Contains classes for the backward smooth relu layer
 * @ingroup smoothrelu_layers
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.smoothrelu;

import com.intel.daal.utils.*;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.Tensor;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__SMOOTHRELU__SMOOTHRELUBACKWARDINPUT"></a>
 * @brief Input object for the backward smoothrelu layer
 */
public final class SmoothreluBackwardInput extends com.intel.daal.algorithms.neural_networks.layers.BackwardInput {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public SmoothreluBackwardInput(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Sets the input object of the backward smoothrelu layer
     * @param id    Identifier of the input object
     * @param val   Value of the input object
     */
    public void set(SmoothreluLayerDataId id, Tensor val) {
        if (id == SmoothreluLayerDataId.auxData) {
            cSetInput(cObject, id.getValue(), val.getCObject());
        }
        else {
            throw new IllegalArgumentException("Incorrect SmoothreluBackwardInputId");
        }
    }

    /**
     * Returns the input object of the backward smoothrelu layer
     * @param id Identifier of the input object
     * @return   Input object that corresponds to the given identifier
     */
    public Tensor get(SmoothreluLayerDataId id) {
        if (id == SmoothreluLayerDataId.auxData) {
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
