/* file: LogisticCrossBackwardInput.java */
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
 * @defgroup logistic_cross_backward Backward Logistic Cross-entropy Layer
 * @brief Contains classes for the backward logistic cross-entropy layer
 * @ingroup logistic_cross
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.logistic_cross;

import com.intel.daal.utils.*;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.Tensor;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LOGISTIC_CROSS__LOGISTICCROSSBACKWARDINPUT"></a>
 * @brief Input object for the backward logistic cross-entropy layer
 */
public final class LogisticCrossBackwardInput extends com.intel.daal.algorithms.neural_networks.layers.loss.LossBackwardInput {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public LogisticCrossBackwardInput(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Sets the input object of the backward logistic cross-entropy layer
     * @param id    Identifier of the input object
     * @param val   Value of the input object
     */
    public void set(LogisticCrossLayerDataId id, Tensor val) {
        if (id == LogisticCrossLayerDataId.auxData || id == LogisticCrossLayerDataId.auxGroundTruth) {
            cSetInput(cObject, id.getValue(), val.getCObject());
        }
        else {
            throw new IllegalArgumentException("Incorrect LogisticCrossLayerDataId");
        }
    }

    /**
     * Returns the input object of the backward logistic cross-entropy layer
     * @param id Identifier of the input object
     * @return   Input object that corresponds to the given identifier
     */
    public Tensor get(LogisticCrossLayerDataId id) {
        if (id == LogisticCrossLayerDataId.auxData || id == LogisticCrossLayerDataId.auxGroundTruth) {
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
