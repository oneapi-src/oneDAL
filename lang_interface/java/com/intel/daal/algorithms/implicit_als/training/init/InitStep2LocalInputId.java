/* file: InitStep2LocalInputId.java */
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
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__INIT__INITSTEP2LOCALINPUTID"></a>
 * @brief Available identifiers of input objects for the implicit ALS initialization algorithm
 *        in the second step of the distributed processing mode
 */
public final class InitStep2LocalInputId {
    private int _value;

    /**
     * Constructs the local input object identifier using the provided value
     * @param value     Value corresponding to the local input object identifier
     */
    public InitStep2LocalInputId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the local input object identifier
     * @return Value corresponding to the local input object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int InputOfStep2FromStep1 = 0;

    /**
     * Partial results of the implicit ALS initialization algorithm computed in the first step
     * and to be transferred to the second step of the distributed initialization algorithm
     */
    public static final InitStep2LocalInputId inputOfStep2FromStep1 = new InitStep2LocalInputId(InputOfStep2FromStep1);
}
/** @} */
