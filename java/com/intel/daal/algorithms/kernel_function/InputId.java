/* file: InputId.java */
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
 * <a name="DAAL-CLASS-ALGORITHMS__KERNEL_FUNCTION__INPUTID"></a>
 * @brief Available identifiers of input objects for the kernel function algorithm
 */
public final class InputId {
    private int _value;

    /**
     * Constructs the input object identifier using the provided value
     * @param value     Value corresponding to the input object identifier
     */
    public InputId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the input object identifier
     * @return Value corresponding to the input object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int XValue = 0;
    private static final int YValue = 1;

    public static final InputId X = new InputId(XValue); /*!< %Input left data table */
    public static final InputId Y = new InputId(YValue); /*!< %Input right data table */
}
/** @} */
