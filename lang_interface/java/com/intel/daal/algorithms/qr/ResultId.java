/* file: ResultId.java */
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

package com.intel.daal.algorithms.qr;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__QR__RESULTID"></a>
 * @brief Available types of the results of the QR decomposition algorithm
 */
public final class ResultId {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    private int _value;

    public ResultId(int value) {
        _value = value;
    }

    public int getValue() {
        return _value;
    }

    private static final int matrixQId = 0;
    private static final int matrixRId = 1;

    public static final ResultId matrixQ = new ResultId(matrixQId); /*!< Orthogonal Matrix Q */
    public static final ResultId matrixR = new ResultId(matrixRId); /*!< Upper Triangular Matrix R */
}
