/* file: ResultId.java */
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
 * @ingroup kernel_function
 * @{
 */
package com.intel.daal.algorithms.kernel_function;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KERNEL_FUNCTION__RESULTID"></a>
 * @brief Available result identifiers for the kernel function algorithm
 */
public final class ResultId {
    private int _value;

    /**
     * Constructs the result object identifier using the provided value
     * @param value     Value corresponding to the result object identifier
     */
    public ResultId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the result object identifier
     * @return Value corresponding to the result object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int ValuesValue = 0;

    /*!< Table with the result of kernel function computation*/
    public static final ResultId values = new ResultId(ValuesValue);
}
/** @} */
