/* file: Parameter.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
 * @ingroup em_gmm_compute
 * @{
 */
package com.intel.daal.algorithms.em_gmm;

import com.intel.daal.utils.*;
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
        LibUtils.loadLibrary();
    }

    Parameter(DaalContext context, long cParameter) {
        super(context);
        this.cObject = cParameter;
    }

    /**
     * Retrieves the number of components in the Gaussian mixture model
     * @return Number of components
     */
    public long getNComponents() {
        return cGetNComponents(this.cObject);
    }

    /**
     * Retrieves the maximal number of iterations
     * @return Maximal number of iterations
     */
    public long getMaxIterations() {
        return cGetMaxIterations(this.cObject);
    }

    /**
     * Retrieves the threshold for the termination of the algorithm
     * @return Threshold for the termination of the algorithm
     */
    public double getAccuracyThreshold() {
        return cGetAccuracyThreshold(this.cObject);
    }

    /**
     * Retrieves the factor for covariance regularization value in case of ill-conditional data
     * @return Factor for covariance regularization value in case of ill-conditional data
     */
    public double getRegularizationFactor() {
        return cGetRegularizationFactor(this.cObject);
    }

    /**
    * Sets the number of components in the Gaussian mixture model
    * @param nComponents Number of components in the Gaussian mixture model
    */
    public void setNComponents(long nComponents) {
        cSetNComponents(this.cObject, nComponents);
    }

    /**
     * Sets the maximal number of iterations
     * @param maxIterations Maximal number of iterations
     */
    public void setMaxIterations(long maxIterations) {
        cSetMaxIterations(this.cObject, maxIterations);
    }

    /**
     * Sets the threshold for the termination of the algorithm
     * @param accuracyThreshold Threshold for the termination of the algorithm
     */
    public void setAccuracyThreshold(double accuracyThreshold) {
        cSetAccuracyThreshold(this.cObject, accuracyThreshold);
    }

    /**
     * Sets the factor for covariance regularization value in case of ill-conditional data
     * @param regularizationFactor Factor for covariance regularization value in case of ill-conditional data
     */
    public void setRegularizationFactor(double regularizationFactor) {
        cSetRegularizationFactor(this.cObject, regularizationFactor);
    }

    /**
     * Sets the identifier of covariance type in the EM for GMM algorithm
     * @param covarianceStorage identifier of covariance type in the EM for GMM algorithm
     */
    public void setCovarianceStorage(CovarianceStorageId covarianceStorage) {
        cSetCovarianceStorage(this.cObject, covarianceStorage.getValue());
    }

    /**
    * Retrieves identifier of covariance type in the EM for GMM algorithm
    * @return identifier of covariance type in the EM for GMM algorithm
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

    private native long cGetNComponents(long parameterAddress);

    private native long cGetMaxIterations(long parameterAddress);

    private native double cGetAccuracyThreshold(long parameterAddress);

    private native double cGetRegularizationFactor(long parameterAddress);

    private native void cSetNComponents(long parameterAddress, long nComponents);

    private native void cSetMaxIterations(long parameterAddress, long maxIterations);

    private native void cSetAccuracyThreshold(long parameterAddress, double accuracyThreshold);

    private native void cSetRegularizationFactor(long parameterAddress, double regularizationFactor);

    private native void cSetCovarianceStorage(long parameterAddress, int covarianceStorageValue);

    private native int cGetCovarianceStorage(long parameterAddress);
}
/** @} */
