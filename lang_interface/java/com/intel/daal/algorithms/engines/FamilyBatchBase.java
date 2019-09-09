/* file: FamilyBatchBase.java */
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
 * @defgroup engines Engines
 * @brief Contains classes for engines
 * @ingroup analysis
 * @{
 */
/**
 * @brief Contains classes for the engines
 */
package com.intel.daal.algorithms.engines;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__ENGINES__FAMILYBATCHBASE"></a>
 * @brief Class representing engines
 *
 * @par References
 *      - Input class
 */
public abstract class FamilyBatchBase extends com.intel.daal.algorithms.engines.BatchBase {
    public Input input;     /*!< %Input of the engine */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs engine algorithm
     * @param context Context to manage the engine
     */
    public FamilyBatchBase(DaalContext context) {
        super(context);
    }

    void add(long numberOfStreams)
    {
        cAdd(numberOfStreams);
    }

    long getNumberOfStreams()
    {
        return cGetNumberOfStreams();
    }

    long getMaxNumberOfStreams()
    {
        return cGetMaxNumberOfStreams();
    }

    private native void cAdd(long numberOfStreams);
    private native long cGetNumberOfStreams();
    private native long cGetMaxNumberOfStreams();
}
/** @} */
