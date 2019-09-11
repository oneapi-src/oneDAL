/* file: MaximumPooling1dBackwardResult.java */
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
 * @ingroup maximum_pooling1d_backward
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.maximum_pooling1d;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__MAXIMUM_POOLING1D__MAXIMUMPOOLING1DBACKWARDRESULT"></a>
 * @brief Provides methods to access results obtained with the compute() method of the backward one-dimensional maximum pooling layer
 */
public class MaximumPooling1dBackwardResult extends com.intel.daal.algorithms.neural_networks.layers.pooling1d.Pooling1dBackwardResult {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the backward one-dimensional maximum pooling layer
     * @param context   Context to manage the backward one-dimensional maximum pooling layer
     */
    public MaximumPooling1dBackwardResult(DaalContext context) {
        super(context);
        this.cObject = cNewResult();
    }

    public MaximumPooling1dBackwardResult(DaalContext context, long cObject) {
        super(context, cObject);
    }

    private native long cNewResult();
}
/** @} */
