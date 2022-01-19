/* file: InitMethod.java */
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

import com.intel.daal.utils.*;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__EM_GMM__INIT__INITMETHOD"></a>
 * @brief Available methods for computing initial values for the EM for GMM algorithm
 */
public final class InitMethod {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    private int _value;

    /**
     * Constructs the initialization method object using the provided value
     * @param value     Value corresponding to the initialization method object
     */
    public InitMethod(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the initialization method object
     * @return Value corresponding to the initialization method object
     */
    public int getValue() {
        return _value;
    }

    private static final int DefaultDenseValue = 0;

    public static final InitMethod defaultDense = new InitMethod(DefaultDenseValue); /*!< Default method */
}
/** @} */
