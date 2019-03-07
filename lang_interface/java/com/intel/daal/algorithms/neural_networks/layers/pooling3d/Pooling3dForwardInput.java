/* file: Pooling3dForwardInput.java */
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
 * @defgroup pooling3d_forward Forward Three-dimensional Pooling Layer
 * @brief Contains classes for the forward three-dimensional (3D) pooling layer
 * @ingroup pooling3d
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.pooling3d;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__POOLING3D__POOLING3DFORWARDINPUT"></a>
 * @brief %Input object for the forward pooling layer
 */
public class Pooling3dForwardInput extends com.intel.daal.algorithms.neural_networks.layers.ForwardInput {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public Pooling3dForwardInput(DaalContext context, long cObject) {
        super(context, cObject);
    }
}
/** @} */
