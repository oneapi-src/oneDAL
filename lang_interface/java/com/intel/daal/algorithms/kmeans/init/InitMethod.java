/* file: InitMethod.java */
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
 * @ingroup kmeans_init
 * @{
 */
package com.intel.daal.algorithms.kmeans.init;

import com.intel.daal.utils.*;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__INIT__INITMETHOD"></a>
 * @brief Methods of computing initial clusters for the K-Means algorithm
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

    private static final int DeterministicDenseValue = 0;
    private static final int RandomDenseValue        = 1;
    private static final int PlusPlusDenseValue      = 2;
    private static final int ParallelPlusDenseValue  = 3;
    private static final int DeterministicCSRValue   = 4;
    private static final int RandomCSRValue          = 5;
    private static final int PlusPlusCSRValue        = 6;
    private static final int ParallelPlusCSRValue    = 7;

    public static final InitMethod defaultDense       = new InitMethod(DeterministicDenseValue); /*!< Default: uses first nClusters points as
                                                                                                      initial clusters */
    public static final InitMethod deterministicDense = new InitMethod(DeterministicDenseValue); /*!< Synonym of deterministicDense */
    public static final InitMethod randomDense        = new InitMethod(RandomDenseValue);        /*!< Uses random nClusters points as initial
                                                                                                      clusters */
    public static final InitMethod plusPlusDense      = new InitMethod(PlusPlusDenseValue);      /*!< Kmeans++ algorithm by Arthur
                                                                                                      and Vassilvitskii (2007)*/
    public static final InitMethod parallelPlusDense  = new InitMethod(ParallelPlusDenseValue);  /*!< Kmeans|| algorithm: scalable Kmeans++
                                                                                                      by Bahmani et al. (2012) */
    public static final InitMethod deterministicCSR   = new InitMethod(DeterministicCSRValue);   /*!< Uses first nClusters points as initial
                                                                                                      clusters for data in a CSR numeric table */
    public static final InitMethod randomCSR          = new InitMethod(RandomCSRValue);          /*!< Uses random nClusters points as initial
                                                                                                      clusters for data in a CSR numeric table */
    public static final InitMethod plusPlusCSR        = new InitMethod(PlusPlusCSRValue);        /*!< Kmeans++ algorithm by Arthur
                                                                                                      and Vassilvitskii (2007)*/
    public static final InitMethod parallelPlusCSR    = new InitMethod(ParallelPlusCSRValue);    /*!< Kmeans|| algorithm: scalable Kmeans++
                                                                                                      by Bahmani et al. (2012) */
}
/** @} */
