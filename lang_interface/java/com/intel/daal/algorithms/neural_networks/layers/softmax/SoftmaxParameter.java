/* file: SoftmaxParameter.java */
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
 * @ingroup softmax_layers
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.softmax;

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__SOFTMAX__SOFTMAXPARAMETER"></a>
 * \brief Class that specifies parameters of the softmax layer
 */
public class SoftmaxParameter extends com.intel.daal.algorithms.neural_networks.layers.Parameter {

    /**
     * Constructs the forward softmax layer parameter
     * @param context   Context to manage the forward softmax layer parameter
     */
    public SoftmaxParameter(DaalContext context) {
        super(context);
        cObject = cInit();
    }

    public SoftmaxParameter(DaalContext context, long cParameter) {
        super(context, cParameter);
    }

    /**
     *  Gets the index of the dimension to calculate softmax
     */
    public long getDimension() {
        return cGetDimension(cObject);
    }

    /**
     *  Sets the index of the dimension to calculate softmax
     *  @param dimension   SoftmaxIndex of the dimension to calculate softmax
     */
    public void setDimension(long dimension) {
        cSetDimension(cObject, dimension);
    }

    private native long cInit();
    private native long cGetDimension(long cParameter);
    private native void cSetDimension(long cParameter, long dimension);
}
/** @} */
