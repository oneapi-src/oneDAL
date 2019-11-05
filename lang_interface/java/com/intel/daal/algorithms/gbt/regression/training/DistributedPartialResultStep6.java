/* file: DistributedPartialResultStep6.java */
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
 * @ingroup gbt_distributed
 * @{
 */
package com.intel.daal.algorithms.gbt.regression.training;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.gbt.regression.Model;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__REGRESSION__TRAINING__DISTRIBUTEDPARTIALRESULTSTEP6"></a>
 * @brief Provides methods to access partial results obtained with the compute() method of
 *        model-based training  in the sixth step of the distributed processing mode
 */
public final class DistributedPartialResultStep6 extends com.intel.daal.algorithms.PartialResult {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs a partial result of model-based training from the context
     * @param context Context to manage the memory in the native part of the partial result object
     */
    public DistributedPartialResultStep6(DaalContext context) {
        super(context);
        this.cObject = cNewDistributedPartialResultStep6();
    }

    DistributedPartialResultStep6(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Returns the partial result of model-based training
     * @param id    Identifier of the partial result
     * @return      Result that corresponds to the given identifier
     */
    public Model get(DistributedPartialResultStep6Id id) {
        int idValue = id.getValue();
        if (idValue != DistributedPartialResultStep6Id.partialModel.getValue()) {
            throw new IllegalArgumentException("id unsupported");
        }
        return new Model(getContext(), cGetModel(cObject, idValue));
    }

    private native long cNewDistributedPartialResultStep6();

    private native long cGetModel(long resAddr, int id);
}
/** @} */
