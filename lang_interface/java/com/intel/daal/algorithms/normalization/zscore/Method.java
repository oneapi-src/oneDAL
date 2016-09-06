/* file: Method.java */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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

package com.intel.daal.algorithms.normalization.zscore;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NORMALIZATION__ZSCORE__METHOD"></a>
 * @brief Available methods for Z-score normalization
 */
public final class Method {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    private int _value;

    /**
     * Constructs the method object using the provided identifier
     */
    public Method(int value) {
        _value = value;
    }

    /**
     * Returns the value of the method identifier
     * @return Value of the method identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int   DefaultDense = 0;
    private static final int   SumDense = 1;
    public static final Method defaultDense       = new Method(DefaultDense); /*!< Default: performance-oriented method.
                                                                              Works with all types  of input numeric tables */
    public static final Method sumDense           = new Method(SumDense);    /*!< Precomputed sum: implementation of computation
                                                                              algorithm in the case of a precomputed sum.
                                                                              Works with all types of input numeric tables */
}
