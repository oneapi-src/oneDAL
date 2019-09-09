/* file: ResultId.java */
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

import java.lang.annotation.Native;

import com.intel.daal.utils.*;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__SVD__RESULTID"></a>
 * @brief Available types of results of the SVD algorithm
 */
public final class ResultId {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

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

    @Native private static final int singularValuesId      = 0;
    @Native private static final int leftSingularMatrixId  = 1;
    @Native private static final int rightSingularMatrixId = 2;

    public static final ResultId singularValues      = new ResultId(singularValuesId); /*!< Singular values         */
    public static final ResultId leftSingularMatrix  = new ResultId(
            leftSingularMatrixId);                                                     /*!< Left orthogonal matrix  */
    public static final ResultId rightSingularMatrix = new ResultId(
            rightSingularMatrixId);                                                    /*!< Right orthogonal matrix */
}
/** @} */
