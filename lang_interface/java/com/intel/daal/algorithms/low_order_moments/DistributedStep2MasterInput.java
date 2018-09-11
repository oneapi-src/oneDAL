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
 * @ingroup low_order_moments_distributed
 * @{
 */
package com.intel.daal.algorithms.low_order_moments;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.ComputeStep;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LOW_ORDER_MOMENTS__DISTRIBUTEDSTEP2MASTERINPUT"></a>
 * @brief Input objects for the low order %moments algorithm on the master node.
  */
public final class DistributedStep2MasterInput extends com.intel.daal.algorithms.Input {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public DistributedStep2MasterInput(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Adds partial result to data collection of input objects for the low order moments algorithm in the distributed processing mode
     * @param id            Identifier of input object identifier
     * @param pres          Partial result obtained on the first step of the distributed processing
     */
    public void add(DistributedStep2MasterInputId id, PartialResult pres) {
        cAddInput(cObject, id.getValue(), pres.getCObject());
    }

    private native void cAddInput(long inputAddr, int id, long presAddr);
}
/** @} */
