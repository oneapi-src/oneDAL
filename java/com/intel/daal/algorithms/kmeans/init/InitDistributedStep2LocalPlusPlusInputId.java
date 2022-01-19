/* file: InitDistributedStep2LocalPlusPlusInputId.java */
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
 * @ingroup kmeans_init_distributed
 * @{
 */
package com.intel.daal.algorithms.kmeans.init;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__INIT__INITDISTRIBUTEDSTEP2LOCALPLUSPLUSINPUTID"></a>
 * @brief Available identifiers of input objects for computing initial clusters for the K-Means algorithm
 *        used with plusPlus and parallelPlus methods only on the 2nd step on a local node.
 */
public final class InitDistributedStep2LocalPlusPlusInputId {
    private int _value;

    /**
     * Constructs the initialization input object identifier using the provided value
     * @param value     Value corresponding to the initialization input object identifier
     */
    public InitDistributedStep2LocalPlusPlusInputId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the initialization input object identifier
     * @return Value corresponding to the initialization input object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int inputOfStep2Value = 2;

    /** Numeric table with the new centers calculated by previous steps of initialization algorithm */
    public static final InitDistributedStep2LocalPlusPlusInputId inputOfStep2 = new InitDistributedStep2LocalPlusPlusInputId(
            inputOfStep2Value);
}
/** @} */
