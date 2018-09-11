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
 * @ingroup logitboost
 */
/**
 * @brief Contains classes of the LogitBoost classification algorithm
 */
package com.intel.daal.algorithms.logitboost;

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LOGITBOOST__PARAMETER"></a>
 * @brief Base class for parameters of the LogitBoost training algorithm
 */
public class Parameter extends com.intel.daal.algorithms.boosting.Parameter {

    public Parameter(DaalContext context, long cParameter) {
        super(context, cParameter);
    }

    /**
     * Sets the accuracy of the LogitBoost training algorithm
     * @param accuracyThreshold Accuracy of the LogitBoost training algorithm
     */
    public void setAccuracyThreshold(double accuracyThreshold) {
        cSetAccuracyThreshold(this.cObject, accuracyThreshold);
    }

    /**
     * Retrieves the accuracy of the LogitBoost training algorithm
     * @return Accuracy of the LogitBoost training algorithm
     */
    public double getAccuracyThreshold() {
        return cGetAccuracyThreshold(this.cObject);
    }

    /**
     * Sets the threshold to avoid degenerate cases when calculating weights W
     * @param weightsDegenerateCasesThreshold The threshold
     */
    public void setWeightsDegenerateCasesThreshold(double weightsDegenerateCasesThreshold) {
        cSetWeightsThreshold(this.cObject, weightsDegenerateCasesThreshold);
    }

    /**
     * Retrieves the threshold needed to avoid degenerate cases when calculating weights W
     * @return The threshold
     */
    public double getWeightsDegenerateCasesThreshold() {
        return cGetWeightsThreshold(this.cObject);
    }

    /**
     * Sets the threshold to avoid degenerate cases when calculating responses Z
     * @param responsesDegenerateCasesThreshold The threshold     */
    public void setResponsesDegenerateCasesThreshold(double responsesDegenerateCasesThreshold) {
        cSetResponsesThreshold(this.cObject, responsesDegenerateCasesThreshold);
    }

    /**
     * Retrieves the threshold needed to avoid degenerate cases when calculating responses Z
     * @return The threshold
     */
    public double getResponsesDegenerateCasesThreshold() {
        return cGetResponsesThreshold(this.cObject);
    }

    /**
     * Sets the maximal number of iterations of the LogitBoost training algorithm
     * @param maxIterations Maximal number of iterations
     */
    public void setMaxIterations(long maxIterations) {
        cSetMaxIterations(this.cObject, maxIterations);
    }

    /**
     * Retrieves the maximal number of iterations of the LogitBoost training algorithm
     * @return Maximal number of iterations
     */
    public long getMaxIterations() {
        return cGetMaxIterations(this.cObject);
    }

    private native void cSetAccuracyThreshold(long parAddr, double acc);

    private native double cGetAccuracyThreshold(long parAddr);

    private native void cSetWeightsThreshold(long parAddr, double acc);

    private native double cGetWeightsThreshold(long parAddr);

    private native void cSetResponsesThreshold(long parAddr, double acc);

    private native double cGetResponsesThreshold(long parAddr);

    private native void cSetMaxIterations(long parAddr, long nIter);

    private native long cGetMaxIterations(long parAddr);
}
/** @} */
