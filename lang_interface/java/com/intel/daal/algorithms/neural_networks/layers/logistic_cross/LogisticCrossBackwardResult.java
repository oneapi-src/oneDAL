/* file: LogisticCrossBackwardResult.java */
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
 * @ingroup logistic_cross_backward
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.logistic_cross;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LOGISTIC_CROSS__LOGISTICCROSSBACKWARDRESULT"></a>
 * @brief Provides methods to access results obtained with the compute() method of the backward logistic cross-entropy layer
 */
public class LogisticCrossBackwardResult extends com.intel.daal.algorithms.neural_networks.layers.loss.LossBackwardResult {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the backward logistic cross-entropy layer result
     * @param context   Context to manage the backward logistic cross-entropy layer result
     */
    public LogisticCrossBackwardResult(DaalContext context) {
        super(context);
        this.cObject = cNewResult();
    }

    public LogisticCrossBackwardResult(DaalContext context, long cObject) {
        super(context, cObject);
    }

    private native long cNewResult();
}
/** @} */
