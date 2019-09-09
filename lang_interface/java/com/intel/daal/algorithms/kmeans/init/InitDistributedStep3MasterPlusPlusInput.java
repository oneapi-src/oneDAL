/* file: InitDistributedStep3MasterPlusPlusInput.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
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
