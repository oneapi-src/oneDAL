/* file: DistributedStep2MasterParameter.java */
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
 * @ingroup pca_distributed
 * @{
 */
package com.intel.daal.algorithms.pca;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.ComputeStep;
import com.intel.daal.services.DaalContext;
import com.intel.daal.algorithms.covariance.DistributedIface;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__DISTRIBUTEDSTEP2MASTERPARAMETER"></a>
 * @brief Parameters of the PCA algorithm in the second step of the distributed processing mode
 */
public class DistributedStep2MasterParameter extends BaseParameter {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public DistributedStep2MasterParameter(DaalContext context, long cObject, long algAddr, Precision prec, Method method) {
        super(context, cObject, algAddr, prec, method, ComputeMode.distributed, ComputeStep.step2Master);
        _isCovarianceInitialized = false;
    }

    /**
     * Sets the correlation or variance-covariance matrix algorithm used by the PCA algorithm
     * @param covariance Correlation or variance-covariance matrix algorithm
     */
    public void setCovariance(DistributedIface covariance) {
        _covariance = covariance;
        _isCovarianceInitialized = true;
        cSetCovariance(this.cObject, _covariance.cDistributedIface, _method.getValue(), _cmode.getValue(), _step.getValue(), _prec.getValue());
    }

    private DistributedIface _covariance;
    private boolean _isCovarianceInitialized;
}
/** @} */
