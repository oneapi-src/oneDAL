/* file: MaximumPooling3dForwardInput.java */
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
 * @defgroup maximum_pooling3d_forward Forward Three-dimensional Max Pooling Layer
 * @brief Contains classes for forward maximum 3D pooling layer
 * @ingroup maximum_pooling3d
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.maximum_pooling3d;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__MAXIMUM_POOLING3D__MAXIMUMPOOLING3DFORWARDINPUT"></a>
 * @brief %Input object for the forward three-dimensional maximum pooling layer
 */
public class MaximumPooling3dForwardInput extends com.intel.daal.algorithms.neural_networks.layers.pooling3d.Pooling3dForwardInput {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public MaximumPooling3dForwardInput(DaalContext context, long cObject) {
        super(context, cObject);
    }
}
/** @} */
