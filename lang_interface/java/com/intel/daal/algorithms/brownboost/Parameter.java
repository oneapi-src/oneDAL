/* file: Parameter.java */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/**
 * @brief Contains classes of the BrownBoost classification algorithm
 */
package com.intel.daal.algorithms.brownboost;

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__BROWNBOOST__PARAMETER"></a>
 * @brief Base class for the parameters of the BrownBoost training algorithm
 */
public class Parameter extends com.intel.daal.algorithms.boosting.Parameter {

    public Parameter(DaalContext context, long cParameter) {
        super(context, cParameter);
    }

    /**
     * Sets the accuracy of the BrownBoost training algorithm
     * @param accuracyThreshold Accuracy of the BrownBoost training algorithm
     */
    public void setAccuracyThreshold(double accuracyThreshold) {
        cSetAccuracyThreshold(this.cObject, accuracyThreshold);
    }

    /**
     * Retrieves the accuracy of the BrownBoost training algorithm
     * @return Accuracy of the BrownBoost training algorithm
     */
    public double getAccuracyThreshold() {
        return cGetAccuracyThreshold(this.cObject);
    }

    /**
     * Sets the accuracy threshold for Newton-Raphson iterations in the BrownBoost training algorithm
     * @param newtonRaphsonAccuracyThreshold Accuracy threshold
     */
    public void setNewtonRaphsonAccuracyThreshold(double newtonRaphsonAccuracyThreshold) {
        cSetNrAccuracyThreshold(this.cObject, newtonRaphsonAccuracyThreshold);
    }

    /**
     * Retrieves the accuracy threshold for Newton-Raphson iterations in the BrownBoost training algorithm
     * @return Accuracy threshold
     */
    public double getNewtonRaphsonAccuracyThreshold() {
        return cGetNrAccuracyThreshold(this.cObject);
    }

    /**
     * Sets the threshold to avoid degenerate cases in the BrownBoost training algorithm
     * @param degenerateCasesThreshold The threshold
     */
    public void setDegenerateCasesThreshold(double degenerateCasesThreshold) {
        cSetThr(this.cObject, degenerateCasesThreshold);
    }

    /**
     * Retrieves the threshold needed to avoid degenerate cases in the BrownBoost training algorithm
     * @return The threshold     */
    public double getDegenerateCasesThreshold() {
        return cGetThr(this.cObject);
    }

    /**
     * Sets the maximal number of iterations of the BrownBoost training algorithm
     * @param maxIterations Maximal number of iterations
     */
    public void setMaxIterations(long maxIterations) {
        cSetMaxIterations(this.cObject, maxIterations);
    }

    /**
     * Retrieves the maximal number of iterations of the BrownBoost training algorithm
     * @return Maximal number of iterations
     */
    public long getMaxIterations() {
        return cGetMaxIterations(this.cObject);
    }

    /**
     * Sets the maximal number of Newton-Raphson iterations in the BrownBoost training algorithm
     * @param newtonRaphsonMaxIterations Maximal number of Newton-Raphson iterations     */
    public void setNewtonRaphsonMaxIterations(long newtonRaphsonMaxIterations) {
        cSetNrMaxIterations(this.cObject, newtonRaphsonMaxIterations);
    }

    /**
     * Retrieves the maximal number of Newton-Raphson iterations in the BrownBoost training algorithm
     * @return Maximal number of Newton-Raphson iterations
     */
    public long getNewtonRaphsonMaxIterations() {
        return cGetNrMaxIterations(this.cObject);
    }

    private native void cSetAccuracyThreshold(long parAddr, double acc);

    private native double cGetAccuracyThreshold(long parAddr);

    private native void cSetNrAccuracyThreshold(long parAddr, double acc);

    private native double cGetNrAccuracyThreshold(long parAddr);

    private native void cSetThr(long parAddr, double acc);

    private native double cGetThr(long parAddr);

    private native void cSetMaxIterations(long parAddr, long nIter);

    private native long cGetMaxIterations(long parAddr);

    private native void cSetNrMaxIterations(long parAddr, long nIter);

    private native long cGetNrMaxIterations(long parAddr);

}
