/* file: InitDistributedStep1LocalInput.java */
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
 * @ingroup implicit_als_init_distributed
 * @{
 */
package com.intel.daal.algorithms.implicit_als.training.init;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__INIT__INITDISTRIBUTEDSTEP1LOCALINPUT"></a>
 * @brief %Input objects for the implicit ALS initialization algorithm in the second step
 *        of the distributed processing mode
 */

public final class InitDistributedStep1LocalInput extends InitInput {

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public InitDistributedStep1LocalInput(DaalContext context, long cAlgorithm, Precision prec, InitMethod method) {
        super(context);
        this.cObject = cGetInput(cAlgorithm, prec.getValue(), method.getValue());
    }

    private native long cGetInput(long cAlgorithm, int prec, int method);
}
/** @} */
