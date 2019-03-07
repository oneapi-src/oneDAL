/* file: DistributedStep2MasterInput.java */
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
 * @ingroup linear_regression_distributed
 * @{
 */
package com.intel.daal.algorithms.linear_regression.training;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LINEAR_REGRESSION__TRAINING__DISTRIBUTEDSTEP2MASTERINPUT"></a>
 * @brief Input object for linear regression model-based training in the second step of the distributed processing mode
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
     * Adds an input object on the master node for linear regression model-based training
     * @param id      Identifier of the input object
     * @param val     Value of the input object
     */
    public void add(MasterInputId id, PartialResult val) {
        if (id != MasterInputId.partialModels) {
            throw new IllegalArgumentException("Incorrect MasterInputId");
        }
        cAddInput(this.cObject, id.getValue(), val.getCObject());
    }

    private native void cAddInput(long cObject, int id, long presAddr);
}
/** @} */
