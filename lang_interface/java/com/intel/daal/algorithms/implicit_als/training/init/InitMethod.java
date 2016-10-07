/* file: InitMethod.java */
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

package com.intel.daal.algorithms.implicit_als.training.init;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__INIT__INITMETHOD"></a>
 * @brief Available methods for computing initial values for the implicit ALS training algorithm
 */
public final class InitMethod {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    private int _value;

/**
 * Constructs the InitMethod object using the provided identifier
 */
    public InitMethod(int value) {
        _value = value;
    }

    /**
     * Returns the value of the input object
     * @return      Value of the input object
     */
    public int getValue() {
        return _value;
    }

    private static final int defaultDenseId = 0;
    private static final int fastCSRId      = 1;

    /** Method for initializing the implicit ALS algorithm */
    public static final InitMethod defaultDense = new InitMethod(
            defaultDenseId);                                     /*!< Default: initialization method for input data stored
                                                                    in the dense format */
    /** Method for initializing the implicit ALS algorithm */
    public static final InitMethod fastCSR      = new InitMethod(
            fastCSRId);                                          /*!< Initialization method for input data stored
                                                                    in the compressed sparse row (CSR) format */
}
