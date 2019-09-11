/* file: LossBackwardInput.java */
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
 * @defgroup loss_backward Backward Loss Layer
 * @brief Contains classes for the backward loss layer
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
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LOSS__LOSSBACKWARDINPUT"></a>
 * @brief Input object for the backward loss layer
 */
public class LossBackwardInput extends com.intel.daal.algorithms.neural_networks.layers.BackwardInput {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public LossBackwardInput(DaalContext context, long cObject) {
        super(context, cObject);
    }
}
/** @} */
