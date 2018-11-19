/* file: InitDistributedStep3MasterPlusPlusInput.java */
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
 * @ingroup kmeans_init_distributed
 * @{
 */
package com.intel.daal.algorithms.kmeans.init;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;
import com.intel.daal.data_management.data.NumericTable;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__INIT__INITDISTRIBUTEDSTEP3MASTERPLUSPLUSINPUT"></a>
 * @brief Input objects for computing initial centroids for the K-Means algorithm
*         used with plusPlus and parallelPlus methods only on the 3rd step on a master node.
 */
public final class InitDistributedStep3MasterPlusPlusInput extends com.intel.daal.algorithms.Input {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public InitDistributedStep3MasterPlusPlusInput(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Add an input object for computing initial centroids for the K-Means algorithm
     * in the 3rd step in the distributed processing mode
     * @param id    Identifier of the input object
     * @param key   Identifier of the node this object comes from
     * @param val   Object that corresponds to the given identifier
     */

    public void add(InitDistributedStep3MasterPlusPlusInputId id, int key, NumericTable val) {
        cAddInput(cObject, id.getValue(), key, val.getCObject());
    }

    private native void cAddInput(long inputAddr, int id, int key, long ntAddr);
}
/** @} */
