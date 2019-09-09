/* file: PruningId.java */
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
* @defgroup decision_tree Decision tree
 * @ingroup training_and_prediction
 * @{
 */

package com.intel.daal.algorithms.decision_tree;

import com.intel.daal.utils.*;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__DECISION_TREE__TRAINING__PRUNINGID"></a>
 * @brief Pruning method for Decision tree algorithm
 */
public final class PruningId {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    private int _value;

    /**
     * Constructs the pruning object identifier using the provided value
     * @param value     Value corresponding to the pruning object identifier
     */
    public PruningId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the pruning object identifier
     * @return Value corresponding to the pruning object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int noneId                = 0;
    private static final int reducedErrorPruningId = 1;

    public static final PruningId none                = new PruningId(noneId);                /*!< Do not prune */
    public static final PruningId reducedErrorPruning = new PruningId(reducedErrorPruningId); /*!< Reduced error pruning */
}
/** @} */
