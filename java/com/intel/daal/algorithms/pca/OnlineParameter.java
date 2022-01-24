/* file: OnlineParameter.java */
/*******************************************************************************
* Copyright 2014-2022 Intel Corporation
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
 * @ingroup pca_online
 * @{
 */
package com.intel.daal.algorithms.pca;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.services.DaalContext;
import com.intel.daal.algorithms.covariance.OnlineImpl;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__ONLINEPARAMETER"></a>
 * @brief Parameters of the PCA algorithm in the online processing mode
 */
public class OnlineParameter extends BaseParameter {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public OnlineParameter(DaalContext context, long cObject, long algAddr, Precision prec, Method method) {
        super(context, cObject, algAddr, prec, method, ComputeMode.online);
        _isCovarianceInitialized = false;
    }

    /**
     * Sets the correlation or variance-covariance matrix algorithm used by the PCA algorithm
     * @param covariance Correlation or variance-covariance matrix algorithm
     */
    public void setCovariance(OnlineImpl covariance) {
        _covariance = covariance;
        _isCovarianceInitialized = true;
        cSetCovariance(this.cObject, _covariance.cOnlineImpl, _method.getValue(), _cmode.getValue(), _step.getValue(),
                       _prec.getValue());
    }

    private OnlineImpl _covariance;
    private boolean _isCovarianceInitialized;
}
/** @} */
