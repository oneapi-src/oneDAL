/* file: PreluBackwardInput.java */
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
 * @defgroup prelu_backward Backward Parametric Rectifier Linear Unit (pReLU) Layer
 * @brief Contains classes for the backward prelu layer
 * @ingroup prelu
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.prelu;

import com.intel.daal.utils.*;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.Tensor;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__PRELU__PRELUBACKWARDINPUT"></a>
 * @brief Input object for the backward prelu layer
 */
public final class PreluBackwardInput extends com.intel.daal.algorithms.neural_networks.layers.BackwardInput {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public PreluBackwardInput(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Sets the input object of the backward prelu layer
     * @param id    Identifier of the input object
     * @param val   Value of the input object
     */
    public void set(PreluLayerDataId id, Tensor val) {
        if (id == PreluLayerDataId.auxData || id == PreluLayerDataId.auxWeights) {
            cSetInput(cObject, id.getValue(), val.getCObject());
        }
        else {
            throw new IllegalArgumentException("Incorrect PreluBackwardInputId");
        }
    }

    /**
     * Returns the input object of the backward prelu layer
     * @param id Identifier of the input object
     * @return   Input object that corresponds to the given identifier
     */
    public Tensor get(PreluLayerDataId id) {
        if (id == PreluLayerDataId.auxData || id == PreluLayerDataId.auxWeights) {
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
