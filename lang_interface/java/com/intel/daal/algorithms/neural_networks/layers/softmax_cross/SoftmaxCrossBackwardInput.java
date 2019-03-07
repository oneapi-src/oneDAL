/* file: SoftmaxCrossBackwardInput.java */
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
 * @defgroup softmax_cross_backward Backward Softmax Cross-entropy Layer
 * @brief Contains classes for the backward softmax cross-entropy layer
 * @ingroup softmax_cross
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.softmax_cross;

import com.intel.daal.utils.*;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.Tensor;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__SOFTMAX_CROSS__SOFTMAXCROSSBACKWARDINPUT"></a>
 * @brief Input object for the backward softmax cross-entropy layer
 */
public final class SoftmaxCrossBackwardInput extends com.intel.daal.algorithms.neural_networks.layers.loss.LossBackwardInput {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public SoftmaxCrossBackwardInput(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Sets the input object of the backward softmax cross-entropy layer
     * @param id    Identifier of the input object
     * @param val   Value of the input object
     */
    public void set(SoftmaxCrossLayerDataId id, Tensor val) {
        if (id == SoftmaxCrossLayerDataId.auxProbabilities || id == SoftmaxCrossLayerDataId.auxGroundTruth) {
            cSetInput(cObject, id.getValue(), val.getCObject());
        }
        else {
            throw new IllegalArgumentException("Incorrect SoftmaxCrossLayerDataId");
        }
    }

    /**
     * Returns the input object of the backward softmax cross-entropy layer
     * @param id Identifier of the input object
     * @return   Input object that corresponds to the given identifier
     */
    public Tensor get(SoftmaxCrossLayerDataId id) {
        if (id == SoftmaxCrossLayerDataId.auxProbabilities || id == SoftmaxCrossLayerDataId.auxGroundTruth) {
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
