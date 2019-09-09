/* file: InitDistributedStep5MasterPlusPlusInput.java */
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
import com.intel.daal.data_management.data.SerializableBase;
import com.intel.daal.data_management.data.Factory;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__INIT__INITDISTRIBUTEDSTEP5MASTERPLUSPLUSINPUT"></a>
 * @brief Input objects for computing initial centroids for the K-Means algorithm
*         used with plusPlus and parallelPlus methods only on the 5th step on a master node.
 */
public final class InitDistributedStep5MasterPlusPlusInput extends com.intel.daal.algorithms.Input {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public InitDistributedStep5MasterPlusPlusInput(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Add an input object for computing initial centroids for the K-Means algorithm
     * in the 5th step in the distributed processing mode
     * @param id    Identifier of the input object
     * @param val   Object that corresponds to the given identifier
     */

    public void add(InitDistributedStep5MasterPlusPlusInputId id, NumericTable val) {
        cAddInput(cObject, id.getValue(), val.getCObject());
    }

    /**
     * Returns an input object for computing initial centroids for the K-Means algorithm
     * in the 5th step in the distributed processing mode
     * @param id    Identifier of the input object
     */

    public SerializableBase get(InitDistributedStep5MasterPlusPlusInputDataId id) {
        if (id != InitDistributedStep5MasterPlusPlusInputDataId.inputOfStep5FromStep3) {
            throw new IllegalArgumentException("id unsupported");
        }
        long addr = cGetInput(getCObject(), id.getValue());
        if(addr == 0)
            return null;
        return Factory.instance().createObject(getContext(), addr);
    }

    /**
    * Sets an input object for computing initial centroids for the K-Means algorithm
    * @param id   Identifier of the input object
    * @param val  Object that corresponds to the given identifier
    */
    public void set(InitDistributedStep5MasterPlusPlusInputDataId id, SerializableBase val) {
        if (id == InitDistributedStep5MasterPlusPlusInputDataId.inputOfStep5FromStep3) {
            cSetInput(cObject, id.getValue(), val.getCObject());
        } else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    private native void cAddInput(long inputAddr, int id, long ntAddr);
    private native long cGetInput(long inputAddr, int id);
    private native void cSetInput(long inputAddr, int id, long serialAddr);
}
/** @} */
