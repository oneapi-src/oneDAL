/* file: BatchParameter.java */
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
 * @ingroup pca_batch
 * @{
 */
package com.intel.daal.algorithms.pca;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.ComputeStep;
import com.intel.daal.services.DaalContext;
import com.intel.daal.algorithms.covariance.BatchImpl;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__BATCHPARAMETER"></a>
 * @brief Parameters of the PCA algorithm in the batch processing mode
 */
public class BatchParameter extends BaseParameter {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public BatchParameter(DaalContext context, long cObject, long algAddr, Precision prec, Method method) {
        super(context, cObject, algAddr, prec, method, ComputeMode.batch);
        _isCovarianceInitialized = false;
    }

    /**
     * Sets the correlation or variance-covariance matrix algorithm used by the PCA algorithm
     * @param covariance Correlation or variance-covariance matrix algorithm
     */
    public void setCovariance(BatchImpl covariance) {
        _covariance = covariance;
        _isCovarianceInitialized = true;
        cSetCovariance(this.cObject, _covariance.cBatchImpl, _method.getValue(), _cmode.getValue(), _step.getValue(),
                       _prec.getValue());
    }

    /**
     * Gets the number of components for PCA transformation.
     * @return  The the number of components for PCA transformation.
     */
    public long getNumberOfComponents() {
        return cGetNumberOfComponents(this.cObject, _method.getValue());
    }

    /**
     * Sets the the number of components for PCA transformation.
     * @param nComponents The number of components for PCA transformation.
     */
    public void setNumberOfComponents(long nComponents) {
        cSetNumberOfComponents(this.cObject, nComponents, _method.getValue());
    }

    /**
     * Gets the number of components for PCA transformation.
     * @return  The the number of components for PCA transformation.
     */
    public boolean getIsDeterministic() {
        return cGetIsDeterministic(this.cObject, _method.getValue());
    }

    /**
     * Sets the the number of components for PCA transformation.
     * @param isDeterministic The number of components for PCA transformation.
     */
    public void setIsDeterministic(boolean isDeterministic) {
        cSetIsDeterministic(this.cObject, isDeterministic, _method.getValue());
    }

    /**
     * Sets the 64 bit integer flag that indicates the results to compute
     * @param resultsToCompute The 64 bit integer flag that indicates the results to compute
     */
    public void setResultsToCompute(long resultsToCompute) {
        cSetResultsToCompute(this.cObject, resultsToCompute, _method.getValue());
    }

    /**
     * Gets the 64 bit integer flag that indicates the results to compute
     * @return The 64 bit integer flag that indicates the results to compute
     */
    public long getResultsToCompute() {
        return cGetResultsToCompute(this.cObject, _method.getValue());
    }

    private native void cSetResultsToCompute(long parAddr, long resultsToCompute, int method);
    private native long cGetResultsToCompute(long parAddr, int method);

    private native void cSetNumberOfComponents(long cObject, long nComponents, int method);
    private native long cGetNumberOfComponents(long cObject, int method);

    private native void cSetIsDeterministic(long cObject, boolean isDeterministic, int method);
    private native boolean cGetIsDeterministic(long cObject, int method);

    private BatchImpl _covariance;
    private boolean _isCovarianceInitialized;
}
/** @} */
