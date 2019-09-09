/* file: OnlineParameter.java */
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
 * @ingroup covariance_online
 * @{
 */
package com.intel.daal.algorithms.covariance;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__ONLINEPARAMETER"></a>
 * @brief Parameters of the correlation or variance-covariance matrix algorithm in the online processing mode
 */
public class OnlineParameter extends Parameter {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public long cAlgorithm;

    /**
     * Constructs the parameter of the correlation or variance-covariance matrix algorithm in the online processing mode
     * @param context   Context to manage the parameter of the correlation or variance-covariance matrix algorithm in the online processing mode
     */
    public OnlineParameter(DaalContext context) {
        super(context);
    }

    public OnlineParameter(DaalContext context, long cObject, long cAlgorithm) {
        super(context);
        this.cObject = cObject;
        this.cAlgorithm = cAlgorithm;
    }

    public void setCParameter(long cParameter) {
        this.cObject = cParameter;
        cSetCParameterObject(this.cObject, this.cAlgorithm);
    }

    private native long cInit(long algAddr, int prec, int method, int cmode);

    private native void cSetCParameterObject(long parameterAddr, long algAddr);
}
/** @} */
