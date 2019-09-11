/* file: DistributedStep3LocalInput.java */
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
 * @ingroup svd_distributed
 * @{
 */
package com.intel.daal.algorithms.svd;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.data_management.data.DataCollection;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__SVD__DISTRIBUTEDSTEP3LOCALINPUT"></a>
 * @brief Input objects for the third step of the SVD algorithm in the distributed processing mode
 */
public class DistributedStep3LocalInput extends com.intel.daal.algorithms.Input {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public DistributedStep3LocalInput(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Returns an input object for the SVD algorithm
     * @param id    Identifier of the input object
     * @param val   Value of the input object
     */
    public void set(DistributedStep3LocalInputId id, DataCollection val) throws IllegalArgumentException {
        if (id != DistributedStep3LocalInputId.inputOfStep3FromStep1
                && id != DistributedStep3LocalInputId.inputOfStep3FromStep2) {
            throw new IllegalArgumentException("Incorrect DistributedStep3LocalInputId");
        }

        cSetDataCollection(cObject, id.getValue(), val.getCObject());
    }

    private native void cSetDataCollection(long cInput, int id, long dcAddr);
}
/** @} */
