/* file: InitDistributedPartialResultStep2Id.java */
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
 * @ingroup implicit_als_init_distributed
 * @{
 */
package com.intel.daal.algorithms.implicit_als.training.init;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__INIT__INITDISTRIBUTEDPARTIALRESULTSTEP2ID"></a>
 * @brief Available identifiers of partial results of the implicit ALS initialization algorithm
 *        in the first step of the distributed processing mode
 */
public final class InitDistributedPartialResultStep2Id {
    private int _value;

    /**
     * Constructs the initialization partial result object identifier using the provided value
     * @param value     Value corresponding to the initialization partial result object identifier
     */
    public InitDistributedPartialResultStep2Id(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the initialization partial result object identifier
     * @return Value corresponding to the initialization partial result object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int TransposedData = 2;

    /**
     * Partial results of the implicit ALS initialization algorithm computed in the first step
     * and to be transferred to the second step of the distributed initialization algorithm
     */
    public static final InitDistributedPartialResultStep2Id transposedData = new InitDistributedPartialResultStep2Id(
            TransposedData);
}
/** @} */
