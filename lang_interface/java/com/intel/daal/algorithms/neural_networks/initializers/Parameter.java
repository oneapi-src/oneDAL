/* file: Parameter.java */
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
 * @ingroup initializers
 * @{
 */
package com.intel.daal.algorithms.neural_networks.initializers;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__INITIALIZERS__PARAMETER"></a>
 * @brief Class that specifies parameters of the neural network weights and biases initializer
 */
public class Parameter extends com.intel.daal.algorithms.Parameter {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /** @private */
    public Parameter(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Sets the layer whose weights and biases are initialized by the initializer
     * @param layer whose weights and biases are initialized by the initializer
     */
    public void setLayer(com.intel.daal.algorithms.neural_networks.layers.ForwardLayer layer) {
        cSetLayer(cObject, layer.cObject);
    }

    private native void cSetLayer(long cObject, long cLayerObject);
}
/** @} */
