/* file: BatchBase.java */
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
 * @defgroup distributions Distributions
 * @brief Contains classes for distributions
 * @ingroup analysis
 * @{
 */
/**
 * @brief Contains classes for the distributions
 */
package com.intel.daal.algorithms.distributions;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DISTRIBUTIONS__BATCHBASE"></a>
 * @brief Class representing distributions
 *
 * @par References
 *      - Input class
 */
public abstract class BatchBase extends com.intel.daal.algorithms.AnalysisBatch {
    public Input input;     /*!< %Input of the distribution */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs distribution algorithm
     * @param context Context to manage the distribution
     */
    public BatchBase(DaalContext context) {
        super(context);
    }

    /**
     * Returns the newly allocated distribution with a copy of input objects
     * and parameters of this distribution
     * @param context   Context to manage the distribution
     * @return The newly allocated distribution
     */
    @Override
    public abstract BatchBase clone(DaalContext context);

    protected native long cGetInput(long cAlgorithm);
}
/** @} */
