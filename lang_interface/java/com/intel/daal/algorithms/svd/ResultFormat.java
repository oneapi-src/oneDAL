/* file: ResultFormat.java */
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
 * @ingroup svd
 * @{
 */
package com.intel.daal.algorithms.svd;

import com.intel.daal.utils.*;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__SVD__RESULTFORMAT"></a>
 * @brief Available options to return result matrices
 */
public final class ResultFormat {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    private int _value;

    /**
     * Constructs the result format object using the provided value
     * @param value     Value corresponding to the result format object
     */
    public ResultFormat(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the result format object
     * @return Value corresponding to the result format object
     */
    public int getValue() {
        return _value;
    }

    private static final int notRequiredId          = 0;
    private static final int requiredInPackedFormId = 1;

    /** Matrix is not required */
    public static final ResultFormat notRequired          = new ResultFormat(notRequiredId);
    /** Matrix in the packed format is required */
    public static final ResultFormat requiredInPackedForm = new ResultFormat(requiredInPackedFormId);
}
/** @} */
