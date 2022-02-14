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
 * @ingroup training
 * @{
 */
package com.intel.daal.algorithms.regression.training;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__REGRESSION__TRAINING__TRAININGRESULT"></a>
 * @brief Provides methods to access results obtained with the compute() method of the regression training algorithm in the batch,
 *        online, or distributed processing mode
 */

public class TrainingResult extends com.intel.daal.algorithms.Result {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the result of the regression training algorithm
     * @param context   Context to manage the result of the regression training algorithm
     */
    public TrainingResult(DaalContext context) {
        super(context);
    }

    public TrainingResult(DaalContext context, long cObject) {
        super(context, cObject);
    }
}
/** @} */
