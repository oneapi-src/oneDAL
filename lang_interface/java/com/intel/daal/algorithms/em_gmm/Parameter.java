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

package com.intel.daal.algorithms.em_gmm;

import com.intel.daal.services.DaalContext;

/**
 * @brief Parameters of the EM for GMM algorithm
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__EM_GMM__PARAMETER"></a>
 * @brief Parameters of the EM for GMM algorithm
 */
public class Parameter extends com.intel.daal.algorithms.Parameter {

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    Parameter(DaalContext context, long cParameter) {
        super(context);
        this.cObject = cParameter;
    }

    Parameter(DaalContext context, long algAddr, int prec, int method, int cmode, long nComponents, long maxIterations,
            double accuracyThreshold, com.intel.daal.algorithms.covariance.Batch covariance) {
        super(context);
        initialize(algAddr, prec, method, cmode, nComponents, maxIterations, accuracyThreshold);
    }

    Parameter(DaalContext context, long algAddr, int prec, int method, int cmode, long nComponents, long maxIterations,
            double accuracyThreshold) {
        super(context);
        initialize(algAddr, prec, method, cmode, nComponents, maxIterations, accuracyThreshold);
    }

    private void initialize(long algAddr, int prec, int method, int cmode, long nComponents, long maxIterations,
            double accuracyThreshold) {
        this.cObject = cInit(algAddr, prec, method, cmode, nComponents, maxIterations, accuracyThreshold);
    }

    /**
     * Retrieves the number of components in the Gaussian mixture model
     * @return Number of components
     */
    long getNComponents() {
        return cGetNComponents(this.cObject);
    }

    /**
     * Retrieves the maximal number of iterations
     * @return Maximal number of iterations
     */
    long getMaxIterations() {
        return cGetMaxIterations(this.cObject);
    }

    /**
     * Retrieves the threshold for the termination of the algorithm
     * @return Threshold for the termination of the algorithm
     */
    double getAccuracyThreshold() {
        return cGetAccuracyThreshold(this.cObject);
    }

    /**
    * Sets the number of components in the Gaussian mixture model
    * @param nComponents Number of components in the Gaussian mixture model
    */
    void setNComponents(long nComponents) {
        cSetNComponents(this.cObject, nComponents);
    }

    /**
     * Sets the maximal number of iterations
     * @param maxIterations Maximal number of iterations
     */
    void setMaxIterations(long maxIterations) {
        cSetMaxIterations(this.cObject, maxIterations);
    }

    /**
     * Sets the threshold for the termination of the algorithm
     * @param accuracyThreshold Threshold for the termination of the algorithm
     */
    void setAccuracyThreshold(double accuracyThreshold) {
        cSetAccuracyThreshold(this.cObject, accuracyThreshold);
    }

    private native long cInit(long algAddr, int prec, int method, int cmode, long nComponents, long maxIterations,
            double accuracyThreshold);

    private native long cGetNComponents(long parameterAddress);

    private native long cGetMaxIterations(long parameterAddress);

    private native double cGetAccuracyThreshold(long parameterAddress);

    private native void cSetNComponents(long parameterAddress, long nComponents);

    private native void cSetMaxIterations(long parameterAddress, long maxIterations);

    private native void cSetAccuracyThreshold(long parameterAddress, double accuracyThreshold);
}
