/* file: Pooling3dForwardResult.java */
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
 * @ingroup pooling3d_forward
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.pooling3d;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__POOLING3D__POOLING3DFORWARDRESULT"></a>
 * @brief Class that provides methods to access the result obtained with the compute() method of the forward pooling layer
 */
public class Pooling3dForwardResult extends com.intel.daal.algorithms.neural_networks.layers.ForwardResult {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
    * Constructs the forward three-dimensional pooling layer
    * @param context   Context to manage the forward three-dimensional pooling layer
    */
    public Pooling3dForwardResult(DaalContext context) {
        super(context);
    }

    public Pooling3dForwardResult(DaalContext context, long cObject) {
        super(context, cObject);
    }
}
/** @} */
