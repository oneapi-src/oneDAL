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

package com.intel.daal.algorithms.kmeans.init;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__INIT__INITMETHOD"></a>
 * @brief Methods of computing initial clusters for the K-Means algorithm
 */
public final class InitMethod {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    private int _value;

    public InitMethod(int value) {
        _value = value;
    }

    /**
     * Returns a value corresponding to the identifier of the input object
     * \return Value corresponding to the identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int DeterministicDenseValue = 0;
    private static final int RandomDenseValue        = 1;
    private static final int DeterministicCSRValue   = 2;
    private static final int RandomCSRValue          = 3;

    public static final InitMethod defaultDense       = new InitMethod(DeterministicDenseValue); /*!< Default: uses first nClusters points as
                                                                                                      initial clusters */
    public static final InitMethod deterministicDense = new InitMethod(DeterministicDenseValue); /*!< Synonym of deterministicDense */
    public static final InitMethod randomDense        = new InitMethod(RandomDenseValue);        /*!< Uses random nClusters points as initial
                                                                                                      clusters */
    public static final InitMethod deterministicCSR   = new InitMethod(DeterministicCSRValue);   /*!< Uses first nClusters points as initial
                                                                                                      clusters for data in a CSR numeric table */
    public static final InitMethod randomCSR          = new InitMethod(RandomCSRValue);          /*!< Uses random nClusters points as initial
                                                                                                      clusters for data in a CSR numeric table */
}
