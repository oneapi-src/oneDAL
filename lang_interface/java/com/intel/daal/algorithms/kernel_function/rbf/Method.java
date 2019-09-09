/* file: Method.java */
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
 * @ingroup kernel_function_rbf
 * @{
 */
package com.intel.daal.algorithms.kernel_function.rbf;

import java.lang.annotation.Native;

import com.intel.daal.utils.*;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__KERNEL_FUNCTION__RBF__METHOD"></a>
 * @brief Available methods for RBF kernel function
 */
public final class Method {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    private int _value;

    /**
     * Constructs the method object using the provided value
     * @param value     Value corresponding to the method object
     */
    public Method(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the method object
     * @return Value corresponding to the method object
     */
    public int getValue() {
        return _value;
    }

    @Native private static final int   defaultDenseValue = 0;
    @Native private static final int   fastCSRValue      = 1;

    /** Default method for computing the RBF kernel */
    public static final Method defaultDense = new Method(defaultDenseValue);

    /** Fast: performance-oriented method. Works with Compressed Sparse Rows (CSR) numeric tables */
    public static final Method fastCSR      = new Method(fastCSRValue);
}
/** @} */
