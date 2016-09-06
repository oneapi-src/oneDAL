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

package com.intel.daal.algorithms.normalization.zscore;

import com.intel.daal.services.DaalContext;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.low_order_moments.BatchIface;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NORMALIZATION__ZSCORE__PARAMETER"></a>
 * @brief Parameters of the z-score normalization algorithm
 */
public class Parameter extends com.intel.daal.algorithms.Parameter {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs the parameter of Z-score normalization algorithm
     *
     * @param context    Context to manage the Z-score normalization algorithm
     * @param cObject    Address of C++ parameter
     * @param algAddr    Address of the algorithm
     * @param prec       Precision of computations
     * @param method     Z-score normalization computation method, @ref Method
     * @param cmode      Computation mode
     */
    public Parameter(DaalContext context, long cObject, long algAddr, Precision prec, Method method, ComputeMode cmode) {
        super(context);
        _prec = prec;
        _method = method;
        _cmode = cmode;
        this.cObject = cObject;
    }

    /**
     * Sets the moments algorithm used by z-score normalization algorithm
     * @param moments Low order moments algorithm
     */
    public void setMoments(BatchIface moments) {
        cSetMoments(this.cObject, moments.cBatchIface, _prec.getValue(), _method.getValue(), _cmode.getValue());
    }
    private Method _method;
    private ComputeMode _cmode;
    private Precision _prec;

    private native void cSetMoments(long cObject, long moments, int prec, int method, int cmode);
}
