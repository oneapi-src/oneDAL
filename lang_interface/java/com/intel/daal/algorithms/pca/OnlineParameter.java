/* file: OnlineParameter.java */
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

package com.intel.daal.algorithms.pca;

import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.ComputeStep;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.services.DaalContext;
import com.intel.daal.algorithms.covariance.OnlineIface;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__ONLINEPARAMETER"></a>
 * @brief Parameters of the PCA algorithm in the online processing mode
 */
public class OnlineParameter extends BaseParameter {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    public OnlineParameter(DaalContext context, long cObject, long algAddr, Precision prec, Method method) {
        super(context, cObject, algAddr, prec, method, ComputeMode.online);
        _isCovarianceInitialized = false;
    }

    /**
     * Sets the correlation or variance-covariance matrix algorithm used by the PCA algorithm
     * @param covariance Correlation or variance-covariance matrix algorithm
     */
    public void setCovariance(OnlineIface covariance) {
        _covariance = covariance;
        _isCovarianceInitialized = true;
        cSetCovariance(this.cObject, _covariance.cOnlineIface, _method.getValue(), _cmode.getValue(), _step.getValue(),
                       _prec.getValue());
    }

    private OnlineIface _covariance;
    private boolean _isCovarianceInitialized;

    /**
     * Sets the initialization procedure for specifying initial parameters of the PCA algorithm
     * @param initializationProcedure   Initialization procedure
     */
    public void setInitializationProcedure(InitializationProcedureIface initializationProcedure) {
        _initializationProcedure = initializationProcedure;
        cSetInitializationProcedure(this.cObject, _initializationProcedure.getCObject(), _method.getValue(), _cmode.getValue(), _step.getValue(),
                                    _prec.getValue());
    }

    /**
     * Gets the initialization procedure for specifying initial parameters of the PCA algorithm
     * @return  Initialization procedure
     */
    public InitializationProcedureIface getInitializationProcedure() {
        if (_initializationProcedure == null) {
            if (_method == Method.correlationDense) {
                _initializationProcedure = new DefaultInitializationProcedureCorrelation(_method);
            }
            if (_method == Method.svdDense) {
                _initializationProcedure = new DefaultInitializationProcedureSVD(_method);
            }
        }
        return _initializationProcedure;
    }

    private InitializationProcedureIface _initializationProcedure;
    private native void cSetInitializationProcedure(long cObject, long cInitProcedure, int method, int cmode, int step, int prec);
}
