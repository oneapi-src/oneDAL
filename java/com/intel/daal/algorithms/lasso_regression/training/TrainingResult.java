/* file: TrainingResult.java */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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
 * @ingroup lasso_regression_training
 * @{
 */
package com.intel.daal.algorithms.lasso_regression.training;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.lasso_regression.Model;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LASSO_REGRESSION__TRAINING__TRAININGRESULT"></a>
 * @brief Provides methods to access the result of lasso regression model-based training
 */
public final class TrainingResult extends com.intel.daal.algorithms.Result {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the lasso regression model-based training result
     * @param context   Context to manage lasso regression training result
     */
    public TrainingResult(DaalContext context) {
        super(context);
        cObject = cNewResult();
    }

    public TrainingResult(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Returns the result of lasso regression model-based training
     * @param id    Identifier of the result
     * @return      Result that corresponds to the given identifier
     */
    public Model get(TrainingResultId id) {
        int idValue = id.getValue();
        if (idValue != TrainingResultId.model.getValue()) {
            throw new IllegalArgumentException("id unsupported");
        }
        return new Model(getContext(), cGetModel(cObject, idValue));
    }

    private native long cNewResult();

    private native long cGetModel(long resAddr, int id);

}
/** @} */
