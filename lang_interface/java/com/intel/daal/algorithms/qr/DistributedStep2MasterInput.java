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
 * @ingroup qr_distributed
 * @{
 */
package com.intel.daal.algorithms.qr;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.data_management.data.DataCollection;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__QR__DISTRIBUTEDSTEP2MASTERINPUT"></a>
 * @brief Input objects for the QR decomposition algorithm on the second step in the distributed processing mode
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
     * Adds value of input object to KeyValueDataCollection of the QR decomposition algorithm
     * @param id    Identifier of input object
     * @param key   Key to be used to retrieve data
     * @param val   Parameter value
     */
    public void add(DistributedStep2MasterInputId id, int key, DataCollection val) {
        if (id == DistributedStep2MasterInputId.inputOfStep2FromStep1) {
            cAddDataCollection(cObject, key, val.getCObject());
        }
    }

    private native void cAddDataCollection(long cInput, int key, long dcAddr);
}
/** @} */
