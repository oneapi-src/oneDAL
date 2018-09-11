/* file: BinaryConfusionMatrixParameter.java */
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
 * @ingroup quality_metric_binary
 * @{
 */
/**
 * @brief Contains classes for computing the binary confusion matrix
 */
package com.intel.daal.algorithms.classifier.quality_metric.binary_confusion_matrix;

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__CLASSIFIER__QUALITY_METRIC__BINARY_CONFUSION_MATRIX__BINARYCONFUSIONMATRIXPARAMETER"></a>
 * @brief Base class for the parameters of the classification algorithms
 */
public class BinaryConfusionMatrixParameter extends com.intel.daal.algorithms.Parameter {

    /**
     * Constructs the parameter of the classification algorithms
     * @param context   Context to manage the parameter of the classification algorithms
     */
    public BinaryConfusionMatrixParameter(DaalContext context) {
        super(context);
    }

    public BinaryConfusionMatrixParameter(DaalContext context, long cParameter) {
        super(context, cParameter);
    }

    /**
     *  Sets the parameter of the F-score quality metric
     *  @param beta  Parameter of the F-score
     */
    public void setBeta(double beta) {
        cSetBeta(this.cObject, beta);
    }

    /**
     *  Gets the parameter of the F-score quality metric
     *  @return  Parameter of the F-score
     */
    public double getBeta() {
        return cGetBeta(this.cObject);
    }

    private native void cSetBeta(long parAddr, double beta);

    private native double cGetBeta(long parAddr);
}
/** @} */
