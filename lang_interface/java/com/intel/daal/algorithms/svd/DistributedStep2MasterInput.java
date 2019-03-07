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
 * @ingroup svd_distributed
 * @{
 */
package com.intel.daal.algorithms.svd;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.data_management.data.DataCollection;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__SVD__DISTRIBUTEDSTEP2MASTERINPUT"></a>
 * @brief DistributedStep2MasterInput objects for the SVD algorithm in the batch processing and online processing modes, and the first step in
 * the distributed processing mode
 */
public class DistributedStep2MasterInput extends com.intel.daal.algorithms.Input {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public DistributedStep2MasterInput(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Adds the value of the input object for the SVD algorithm to KeyValueDataCollection
     * @param id    Identifier of the input object
     * @param key   Key to use to retrieve data
     * @param val   Value of the input object
     */
    public void add(DistributedStep2MasterInputId id, int key, DataCollection val) {
        if (id == DistributedStep2MasterInputId.inputOfStep2FromStep1) {
            cAddDataCollection(cObject, key, val.getCObject());
        }
    }

    private native void cAddDataCollection(long cInput, int key, long dcAddr);
}
/** @} */
