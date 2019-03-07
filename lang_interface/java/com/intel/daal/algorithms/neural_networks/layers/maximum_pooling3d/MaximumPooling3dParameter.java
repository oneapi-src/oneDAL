/* file: MaximumPooling3dParameter.java */
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
 * @ingroup maximum_pooling3d
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.maximum_pooling3d;

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__MAXIMUM_POOLING3D__MAXIMUMPOOLING3DPARAMETER"></a>
 * \brief Class that specifies parameters of the three-dimensional maximum pooling layer
 */
public class MaximumPooling3dParameter extends com.intel.daal.algorithms.neural_networks.layers.pooling3d.Pooling3dParameter {
    /** @private */
    public MaximumPooling3dParameter(DaalContext context, long cObject) {
        super(context, cObject);
    }
}
/** @} */
