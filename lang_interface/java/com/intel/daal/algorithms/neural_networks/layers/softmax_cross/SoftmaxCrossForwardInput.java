/* file: SoftmaxCrossForwardInput.java */
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
 * @defgroup softmax_cross_forward Forward Softmax Cross-entropy Layer
 * @brief Contains classes for the forward softmax cross-entropy layer
 * @ingroup softmax_cross
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.softmax_cross;

import com.intel.daal.utils.*;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.Tensor;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__SOFTMAX_CROSS__SOFTMAXCROSSFORWARDINPUT"></a>
 * @brief %Input object for the forward softmax cross-entropy layer
 */
public class SoftmaxCrossForwardInput extends com.intel.daal.algorithms.neural_networks.layers.loss.LossForwardInput {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public SoftmaxCrossForwardInput(DaalContext context, long cObject) {
        super(context, cObject);
    }
}
/** @} */
