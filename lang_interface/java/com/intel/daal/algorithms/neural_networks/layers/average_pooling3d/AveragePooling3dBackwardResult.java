/* file: AveragePooling3dBackwardResult.java */
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
 * @ingroup average_pooling3d_backward
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.average_pooling3d;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__AVERAGE_POOLING3D__AVERAGEPOOLING3DBACKWARDRESULT"></a>
 * @brief Provides methods to access results obtained with the compute() method of the backward three-dimensional average pooling layer
 */
public class AveragePooling3dBackwardResult extends com.intel.daal.algorithms.neural_networks.layers.pooling3d.Pooling3dBackwardResult {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the backward three-dimensional average pooling layer result
     * @param context   Context to manage the backward three-dimensional average pooling layer result
     */
    public AveragePooling3dBackwardResult(DaalContext context) {
        super(context);
        this.cObject = cNewResult();
    }

    public AveragePooling3dBackwardResult(DaalContext context, long cObject) {
        super(context, cObject);
    }

    private native long cNewResult();
}
/** @} */
