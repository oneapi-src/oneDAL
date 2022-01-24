/* file: InitInputId.java */
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
 * @ingroup em_gmm_init
 * @{
 */
package com.intel.daal.algorithms.em_gmm.init;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__EM_GMM__INIT__INITINPUTID"></a>
 * @brief Available identifiers of input objects for the default initialization of the EM for GMM algorithm
 */
public final class InitInputId {
    private int _value;

    /**
     * Constructs the default initialization input object identifier using the provided value
     * @param value     Value corresponding to the default initialization input object identifier
     */
    public InitInputId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the default initialization input object identifier
     * @return Value corresponding to the default initialization input object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int Data = 0;

    public static final InitInputId data = new InitInputId(Data); /*!< %Input data table */
}
/** @} */
