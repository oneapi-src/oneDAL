/* file: VotingMethod.java */
/*******************************************************************************
* Copyright 2014-2021 Intel Corporation
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
 * @ingroup decision_forest_classification_prediction
 * @{
 */
package com.intel.daal.algorithms.decision_forest.classification.prediction;

import java.lang.annotation.Native;

import com.intel.daal.utils.*;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__DECISION_FOREST__CLASSIFICATION__PREDICTION__VOTING__METHOD"></a>
 * @brief Available voting methods for decision forest
 */
public final class VotingMethod {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    private int _value;

    /**
     * Constructs the method object using the provided value
     * @param value     Value corresponding to the method object
     */
    private VotingMethod(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the method object
     * @return Value corresponding to the method object
     */
    public int getValue() {
        return _value;
    }

    @Native private static final int weightedValue   = 0;
    @Native private static final int unweightedValue = 1;

    public static final VotingMethod weighted   = new VotingMethod(weightedValue);           /*!< Weighted method. */
    public static final VotingMethod unweighted = new VotingMethod(unweightedValue);         /*!< Unweighted method. */
}
/** @} */
