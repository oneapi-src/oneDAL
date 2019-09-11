/* file: EltwiseSumParameter.java */
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
 * @ingroup eltwise_sum
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.eltwise_sum;

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__ELTWISE_SUM__ELTWISESUMPARAMETER"></a>
 * \brief Class that specifies parameters of the element-wise sum layer
 */
public class EltwiseSumParameter extends com.intel.daal.algorithms.neural_networks.layers.Parameter {

    /**
     *  Constructs the parameters for the element-wise sum layer
     */
    public EltwiseSumParameter(DaalContext context) {
        super(context);
        cObject = cInit();
    }

    public EltwiseSumParameter(DaalContext context, long cParameter) {
        super(context, cParameter);
    }

    private native long cInit();
}
/** @} */
