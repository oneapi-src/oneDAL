/* file: OnlineParameter.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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
