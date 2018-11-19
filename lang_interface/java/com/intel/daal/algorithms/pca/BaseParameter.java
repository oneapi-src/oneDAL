/* file: BaseParameter.java */
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
 * @defgroup pca Principal Component Analysis
 * @brief Contains classes for computing the results of the principal component analysis (PCA) algorithm
 * @ingroup analysis
 * @{
 */
package com.intel.daal.algorithms.pca;

import com.intel.daal.utils.*;
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
        LibUtils.loadLibrary();
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
/** @} */
