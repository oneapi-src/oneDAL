/* file: LossParameter.java */
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
 * @ingroup loss
 * @{
 */
/**
 * @brief Contains classes for the neural network layers
 */
package com.intel.daal.algorithms.neural_networks.layers.loss;

import com.intel.daal.utils.*;
import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;
import com.intel.daal.algorithms.neural_networks.initializers.InitializerIface;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LOSS__LOSSPARAMETER"></a>
 * @brief Class that specifies parameters of the neural network layer
 */
public class LossParameter extends com.intel.daal.algorithms.neural_networks.layers.Parameter {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the parameter of the loss layer
     * @param context Context to manage the parameter of the loss layer
     */
    public LossParameter(DaalContext context) {
        super(context);
    }

    public LossParameter(DaalContext context, long cParameter) {
        super(context, cParameter);
    }
}
/** @} */
