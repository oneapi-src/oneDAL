/* file: SoftmaxCrossParameter.java */
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
 * @ingroup softmax_cross
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.softmax_cross;

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__SOFTMAX_CROSS__SOFTMAXCROSSPARAMETER"></a>
 * \brief Class that specifies parameters of the softmax cross-entropy layer
 */
public class SoftmaxCrossParameter extends com.intel.daal.algorithms.neural_networks.layers.loss.LossParameter {

    /**
     *  Constructs the parameters for the softmax cross-entropy layer
     */
    public SoftmaxCrossParameter(DaalContext context) {
        super(context);
        cObject = cInit();
    }

    public SoftmaxCrossParameter(DaalContext context, long cParameter) {
        super(context, cParameter);
    }

    /**
     *  Gets the value needed to avoid degenerate cases in logarithm computing
     */
    public double getAccuracyThreshold() {
        return cGetAccuracyThreshold(cObject);
    }

    /**
     *  Sets the value needed to avoid degenerate cases in logarithm computing
     *  @param accuracyThreshold Value needed to avoid degenerate cases in logarithm computing
     */
    public void setAccuracyThreshold(double accuracyThreshold) {
        cSetAccuracyThreshold(cObject, accuracyThreshold);
    }

    /**
     *  Gets the dimension index used to calculate softmax cross-entropy
     */
    public double getDimension() {
        return cGetDimension(cObject);
    }

    /**
     *  Sets the dimension index used to calculate softmax cross-entropy
     *  @param dimension Dimension index used to calculate softmax cross-entropy
     */
    public void setDimension(double dimension) {
        cSetDimension(cObject, dimension);
    }

    private native long   cInit();
    private native double cGetAccuracyThreshold(long cParameter);
    private native void   cSetAccuracyThreshold(long cParameter, double accuracyThreshold);
    private native double cGetDimension(long cParameter);
    private native void   cSetDimension(long cParameter, double dimension);
}
/** @} */
