/* file: InitParameter.java */
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
 * @ingroup em_gmm_init
 * @{
 */
package com.intel.daal.algorithms.em_gmm.init;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;
import com.intel.daal.algorithms.em_gmm.CovarianceStorageId;

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
        LibUtils.loadLibrary();
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
     * @DAAL_DEPRECATED
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
     * @DAAL_DEPRECATED
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

    /**
    * Sets the identifier of covariance type in the EM for GMM initialization algorithm
    * @param covarianceStorage identifier of covariance type in the EM for GMM initialization algorithm
    */
    public void setCovarianceStorage(CovarianceStorageId covarianceStorage) {
        cSetCovarianceStorage(this.cObject, covarianceStorage.getValue());
    }

    /**
    * Retrieves identifier of covariance type in the EM for GMM initialization algorithm
    * @return identifier of covariance type in the EM for GMM initialization algorithm
    */
    public CovarianceStorageId getCovarianceStorage() {
        int id = cGetCovarianceStorage(this.cObject);
        if(id == CovarianceStorageId.diagonal.getValue()) {
            return CovarianceStorageId.diagonal;
        }
        else {
            return CovarianceStorageId.full;
        }
    }

    /**
     * Sets the engine to be used by the EM for GMM initialization algorithm
     * @param engine to be used by the EM for GMM initialization algorithm
     */
    public void setEngine(com.intel.daal.algorithms.engines.BatchBase engine) {
        cSetEngine(cObject, engine.cObject);
    }

    private native void cSetEngine(long cObject, long cEngineObject);

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

    private native void cSetCovarianceStorage(long parameterAddress, int covarianceStorageValue);

    private native int cGetCovarianceStorage(long parameterAddress);
}
/** @} */
