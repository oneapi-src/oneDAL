/* file: PartialResult.java */
/*******************************************************************************
* Copyright 2014-2022 Intel Corporation
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
 * @ingroup ridge_regression_training
 * @{
 */
package com.intel.daal.algorithms.ridge_regression.training;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ridge_regression.Model;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__RIDGE_REGRESSION__TRAINING__PARTIALRESULT"></a>
 * @brief Provides methods to access a partial result obtained with the compute() method of ridge regression model-based training in the online or
 *        distributed processing mode
 */
public final class PartialResult extends com.intel.daal.algorithms.PartialResult {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public PartialResult(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Returns a partial result of ridge regression model-based training
     * @param id    Identifier of the partial result
     * @return      Partial result that corresponds to the given identifier
     */
    public Model get(PartialResultId id) {
        if (id == PartialResultId.model) {
            return new Model(getContext(), cGetModel(getCObject(), PartialResultId.model.getValue()));
        } else {
            return null;
        }
    }

    private native long cGetModel(long resAddr, int id);
}
/** @} */
