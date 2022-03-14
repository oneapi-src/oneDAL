/* file: InitInputId.java */
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
 * @ingroup kmeans_init
 * @{
 */
package com.intel.daal.algorithms.kmeans.init;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__INIT__INITINPUTID"></a>
 * @brief Available identifiers of input objects for computing initial clusters for the K-Means algorithm
 */
public final class InitInputId {
    private int _value;

    /**
     * Constructs the initialization input object identifier using the provided value
     * @param value     Value corresponding to the initialization input object identifier
     */
    public InitInputId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the initialization input object identifier
     * @return Value corresponding to the initialization input object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int DefaultInputId = 0;

    public static final InitInputId data = new InitInputId(DefaultInputId); /*!< %Input data table */
}
/** @} */
