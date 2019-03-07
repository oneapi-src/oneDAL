/* file: LossForwardInput.java */
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
 * @defgroup loss_forward Forward Loss Layer
 * @brief Contains classes for the forward loss layer
 * @ingroup loss
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.loss;

import com.intel.daal.utils.*;
import com.intel.daal.utils.*;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.Tensor;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LOSS__LOSSFORWARDINPUT"></a>
 * @brief %Input object for the forward loss layer
 */
public class LossForwardInput extends com.intel.daal.algorithms.neural_networks.layers.ForwardInput {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public LossForwardInput(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
    * Sets the input object of the forward layer
    * @param id    Identifier of the input object
    * @param val   Value of the input object
    */
    public void set(LossForwardInputId id, Tensor val) {
        if (id == LossForwardInputId.data || id == LossForwardInputId.weights || id == LossForwardInputId.biases|| id == LossForwardInputId.groundTruth) {
            cSetInput(cObject, id.getValue(), val.getCObject());
        }
        else {
            throw new IllegalArgumentException("Incorrect LossForwardInputId");
        }
    }

    /**
     * Returns the input object of the forward layer
     * @param id Identifier of the input object
     * @return   Input object that corresponds to the given identifier
     */
    public Tensor get(LossForwardInputId id) {
        if (id == LossForwardInputId.data || id == LossForwardInputId.weights || id == LossForwardInputId.biases || id == LossForwardInputId.groundTruth) {
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
