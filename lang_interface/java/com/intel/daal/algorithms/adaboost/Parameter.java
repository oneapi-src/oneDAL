/* file: Parameter.java */
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
 * @defgroup adaboost Adaboost Classifier
 * @brief Contains classes for the AdaBoost classification algorithm
 * @ingroup boosting
 */
/**
 * @brief Contains classes for the AdaBoost classification algorithm
 */
package com.intel.daal.algorithms.adaboost;

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__ADABOOST__PARAMETER"></a>
 * @brief AdaBoost algorithm parameters
 */
public class Parameter extends com.intel.daal.algorithms.boosting.Parameter {

    public Parameter(DaalContext context, long cParameter) {
        super(context, cParameter);
    }

    /**
     * Sets the accuracy of the AdaBoost training algorithm
     * @param accuracyThreshold Accuracy of the AdaBoost training algorithm
     */
    public void setAccuracyThreshold(double accuracyThreshold) {
        cSetAccuracyThreshold(this.cObject, accuracyThreshold);
    }

    /**
     * Retrieves the accuracy of the AdaBoost training algorithm
     * @return Accuracy of the AdaBoost training algorithm
     */
    public double getAccuracyThreshold() {
        return cGetAccuracyThreshold(this.cObject);
    }

    /**
     * Sets the maximal number of iterations for the AdaBoost training algorithm
     * @param maxIterations Maximal number of iterations
     */
    public void setMaxIterations(long maxIterations) {
        cSetMaxIterations(this.cObject, maxIterations);
    }

    /**
     * Retrieves the maximal number of iterations for the AdaBoost training algorithm
     * @return Maximal number of iterations
     */
    public long getMaxIterations() {
        return cGetMaxIterations(this.cObject);
    }

    private native void cSetAccuracyThreshold(long parAddr, double acc);

    private native double cGetAccuracyThreshold(long parAddr);

    private native void cSetMaxIterations(long parAddr, long nIter);

    private native long cGetMaxIterations(long parAddr);
}
/** @} */
