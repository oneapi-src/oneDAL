/* file: SplitMethod.java */
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
 * @ingroup gbt
 */
/**
 * @brief Contains classes of the gradient boosted trees algorithm training
 */
package com.intel.daal.algorithms.gbt.training;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__TRAINING__SPLITMETHOD"></a>
 * @brief Split finding method in gradient boosted trees algorithm
 */
public final class SplitMethod {
    private int _value;

    /**
     * Constructs the split method identifier using the provided value
     * @param value     Value corresponding to the split method identifier
     */
    public SplitMethod(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the split method identifier
     * @return Value corresponding to the split method identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int exactId        = 0;
    private static final int inexactId      = 1;
    private static final int defaultSplitId = inexactId;

    public static final SplitMethod exact        = new SplitMethod(exactId);        /*!< Exact greedy method */
    public static final SplitMethod inexact      = new SplitMethod(inexactId);      /*!< Inexact method for splits finding: bucket continuous features to discrete bins */
    public static final SplitMethod defaultSplit = new SplitMethod(defaultSplitId); /*!< Default split finding method */
}
/** @} */
