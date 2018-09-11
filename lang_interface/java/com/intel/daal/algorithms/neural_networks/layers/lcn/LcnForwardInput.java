/* file: LcnForwardInput.java */
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
 * @defgroup lcn_layers_forward Forward Local contrast normalization (LCN) Layer
 * @brief Contains classes for the forward local contrast normalization layer
 * @ingroup lcn_layers
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.lcn;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LCN__LCNFORWARDINPUT"></a>
 * @brief %Input object for the forward local contrast normalization layer
 */
public class LcnForwardInput extends com.intel.daal.algorithms.neural_networks.layers.ForwardInput {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the forward local contrast normalization layer input
     * @param context   Context to manage the forward local contrast normalization layer input
     * @param cObject   Address of C++ forward input
     */
    public LcnForwardInput(DaalContext context, long cObject) {
        super(context, cObject);
    }
}
/** @} */
