/* file: BaseParameter.java */
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
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.ComputeStep;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__BASEPARAMETER"></a>
 * @brief Common parameters of the PCA algorithm
 */
public class BaseParameter extends com.intel.daal.algorithms.Parameter {
    public long _algAddr;
    public Method _method;
    public ComputeMode _cmode;
    public ComputeStep _step;
    public Precision _prec;

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    public BaseParameter(DaalContext context, long cObject, long algAddr, Precision prec, Method method, ComputeMode cmode, ComputeStep step) {
        super(context);
        _algAddr = algAddr;
        _method = method;
        _cmode = cmode;
        _step = step;
        _prec = prec;
        this.cObject = cObject;
    }

    public BaseParameter(DaalContext context, long cObject, long algAddr, Precision prec, Method method, ComputeMode cmode) {
        super(context);
        _algAddr = algAddr;
        _method = method;
        _cmode = cmode;
        _step = ComputeStep.step1Local;
        _prec = prec;
        this.cObject = cObject;
    }

    protected native void cSetCovariance(long cObject, long cCovariance, int method, int cmode, int step, int prec);
}
