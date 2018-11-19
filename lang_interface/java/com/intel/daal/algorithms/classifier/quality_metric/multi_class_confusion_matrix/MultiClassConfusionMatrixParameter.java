/* file: MultiClassConfusionMatrixParameter.java */
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
 * @ingroup quality_metric_multiclass
 * @{
 */
/**
 * @brief Contains classes for computing the multi-class confusion matrix*/
package com.intel.daal.algorithms.classifier.quality_metric.multi_class_confusion_matrix;

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__CLASSIFIER__QUALITY_METRIC__MULTI_CLASS_CONFUSION_MATRIX__MULTICLASSCONFUSIONMATRIXPARAMETER"></a>
 * @brief Base class for the parameters of the multi-class confusion matrix algorithm
 */
public class MultiClassConfusionMatrixParameter extends com.intel.daal.algorithms.Parameter {

    /**
     * Constructs the parameter of the multi-class confusion matrix algorithm
     * @param context   Context to manage the parameter of the multi-class confusion matrix algorithm
     */
    public MultiClassConfusionMatrixParameter(DaalContext context) {
        super(context);
    }

    public MultiClassConfusionMatrixParameter(DaalContext context, long cParameter) {
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
     *  Gets the beta parameter of the F-score quality metric
     *  @return  Parameter of the F-score
     */
    public double getBeta() {
        return cGetBeta(this.cObject);
    }

    /**
     *  Sets the number of classes
     *  @param nClasses  Number of classes
     */
    public void setNClasses(long nClasses) {
        cSetNClasses(this.cObject, nClasses);
    }

    /**
     *  Gets the number of classes
     *  @return  Number of classes
     */
    public long getNClasses() {
        return cGetNClasses(this.cObject);
    }

    private native void cSetNClasses(long parAddr, long nClasses);

    private native long cGetNClasses(long parAddr);

    private native void cSetBeta(long parAddr, double beta);

    private native double cGetBeta(long parAddr);
}
/** @} */
