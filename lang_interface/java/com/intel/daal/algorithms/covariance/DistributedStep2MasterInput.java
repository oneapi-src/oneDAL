/* file: DistributedStep2MasterInput.java */
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
 * @ingroup covariance_distributed
 * @{
 */
package com.intel.daal.algorithms.covariance;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__DISTRIBUTEDSTEP2MASTERINPUT"></a>
 * @brief %Input objects for the correlation or variance-covariance matrix algorithm
 * in the second step of the distributed processing mode
 */
public final class DistributedStep2MasterInput extends com.intel.daal.algorithms.Input {
    public long cAlgorithm;
    public Precision prec;
    public Method                               method;  /*!< Computation method for the algorithm */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public DistributedStep2MasterInput(DaalContext context, long cAlgorithm, Precision prec, Method method) {
        super(context);
        this.cObject = cInit(cAlgorithm, prec.getValue(), method.getValue());

        this.cAlgorithm = cAlgorithm;
        this.prec = prec;
        this.method = method;
    }

    /**
     * Adds a partial result to the end of the data collection of input objects for the correlation or
     * variance-covariance matrix algorithm in the second step of the distributed processing mode
     * @param id            Identifier of the input object
     * @param pres          Partial result obtained in the first step of the distributed processing mode
     */
    public void add(DistributedStep2MasterInputId id, PartialResult pres) {
        cAddInput(cObject, id.getValue(), pres.getCObject());
    }

    public void setCInput(long cInput) {
        this.cObject = cInput;
        cSetCInputObject(this.cObject, this.cAlgorithm, prec.getValue(), method.getValue());
    }

    private native long cInit(long algAddr, int prec, int method);
    private native void cSetCInputObject(long inputAddr, long algAddr, int prec, int method);
    private native void cAddInput(long algAddr, int id, long presAddr);
}
/** @} */
