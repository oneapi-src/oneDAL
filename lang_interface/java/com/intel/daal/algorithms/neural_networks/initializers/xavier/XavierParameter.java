/* file: XavierParameter.java */
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
 * @ingroup initializers_xavier
 * @{
 */
package com.intel.daal.algorithms.neural_networks.initializers.xavier;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__INITIALIZERS__XAVIER__XAVIERPARAMETER"></a>
 * @brief Class that specifies parameters of the neural network weights and biases Xavier initializer
 */
public class XavierParameter extends com.intel.daal.algorithms.neural_networks.initializers.Parameter {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /** @private */
    public XavierParameter(DaalContext context, long cObject) {
        super(context, cObject);
    }
}
/** @} */
