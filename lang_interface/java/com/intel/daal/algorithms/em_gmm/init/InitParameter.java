/* file: InitParameter.java */
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

package com.intel.daal.algorithms.em_gmm.init;

import com.intel.daal.services.DaalContext;

/**
 * @brief Parameters for the default initialization of the EM for GMM algorithm
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__EM_GMM__INIT__INITPARAMETER"></a>
 * @brief Parameters for the default initialization of the EM for GMM algorithm
 */
public class InitParameter extends com.intel.daal.algorithms.Parameter {

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    public InitParameter(DaalContext context, long cInitParameter) {
        super(context);
        this.cObject = cInitParameter;
    }

    public InitParameter(DaalContext context, long algAddr, int prec, int method, int cmode, long nComponents,
            long nTries, long nDepthIterations, double accuracyThreshold) {
        super(context);
        initialize(algAddr, prec, method, cmode, nComponents, nTries, nDepthIterations, accuracyThreshold);
    }

    private void initialize(long algAddr, int prec, int method, int cmode, long nComponents, long nTries,
            long nDepthIterations, double accuracyThreshold) {
        this.cObject = cInit(algAddr, prec, method, cmode, nComponents, nTries, nDepthIterations, accuracyThreshold);
    }

    /**
     * Retrieves the number of components in the Gaussian mixture model
     * @return Number of components
     */
    public long getNComponents() {
        return cGetNComponents(this.cObject);
    }

    /**
     * Retrieves the number of iterations in every short EM run
     * @return Number of iterations in every short EM run
     */
    public long getNDepthIterations() {
        return cGetDepthIterations(this.cObject);
    }

    /**
     * Retrieves the number of trials of short EM runs
     * @return Number of trials of short EM runs
     */
    public long getNTrials() {
        return cGetNTrials(this.cObject);
    }

    /**
     * Retrieves the seed for randomly generating data points to start the initialization of short EM
     * @return Seed for randomly generating data points to start the initialization of short EM
     */
    public long getStartSeed() {
        return cGetStartSeed(this.cObject);
    }

    /**
     * Retrieves the threshold for the termination of the algorithms
     * @return Threshold for the termination of the algorithms
     */
    public double getAccuracyThreshold() {
        return cGetAccuracyThreshold(this.cObject);
    }

    /**
    * Sets the number of components in the Gaussian mixture model.
    * @param nComponents Number of components in the Gaussian mixture model.
    */
    public void setNComponents(long nComponents) {
        cSetNComponents(this.cObject, nComponents);
    }

    /**
     * Sets the number of iterations in every short EM run
     * @param nDepthIterations Number of iterations in every short EM run
     */
    public void setNDepthIterations(long nDepthIterations) {
        cSetNDepthIterations(this.cObject, nDepthIterations);
    }

    /**
     * Sets the number of trials of short EM runs
     * @param nTrials Number of trials of short EM runs
     */
    public void setNTrials(long nTrials) {
        cSetNTrials(this.cObject, nTrials);
    }

    /**
     * Sets the seed for randomly generating data points to start the initialization of short EM
     * @param startSeed Seed for randomly generating data points to start the initialization of short EM
     */
    public void setStartSeed(long startSeed) {
        cSetStartSeed(this.cObject, startSeed);
    }

    /**
     * Sets the threshold for the termination of the algorithm
     * @param accuracyThreshold Threshold for the termination of the algorithm
     */
    public void setAccuracyThreshold(double accuracyThreshold) {
        cSetAccuracyThreshold(this.cObject, accuracyThreshold);
    }

    private native long cInit(long algAddr, int prec, int method, int cmode, long nComponents, long nTries,
            long nDepthIterations, double accuracyThreshold);

    private native long cGetNComponents(long parameterAddress);

    private native long cGetDepthIterations(long parameterAddress);

    private native long cGetNTrials(long parameterAddress);

    private native long cGetStartSeed(long parameterAddress);

    private native double cGetAccuracyThreshold(long parameterAddress);

    private native void cSetNComponents(long parameterAddress, long nComponents);

    private native void cSetNDepthIterations(long parameterAddress, long maxIterations);

    private native void cSetNTrials(long parameterAddress, long maxIterations);

    private native void cSetStartSeed(long parameterAddress, long startSeed);

    private native void cSetAccuracyThreshold(long parameterAddress, double accuracyThreshold);
}
