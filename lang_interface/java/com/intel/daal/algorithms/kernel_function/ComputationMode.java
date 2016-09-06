/* file: ComputationMode.java */
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

package com.intel.daal.algorithms.kernel_function;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KERNEL_FUNCTION__COMPUTATIONMODE"></a>
 * @brief Available modes of kernel function computation
 */
public final class ComputationMode {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    private int _value;

    public ComputationMode(int value) {
        _value = value;
    }

    /**
     * Returns the value of the ComputationMode identifier
     * @return Value of the ComputationMode identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int VectorVector = 0;
    private static final int MatrixVector = 1;
    private static final int MatrixMatrix = 2;

    public static final ComputationMode vectorVector = new ComputationMode(VectorVector); /*!< Computes the kernel function for given feature vectors
                                                                                          Xi and Yj */
    public static final ComputationMode matrixVector = new ComputationMode(MatrixVector); /*!< Computes the kernel function
                                                                                          for all the vectors in the set X and
                                                                                          a given feature vector Yi */
    public static final ComputationMode matrixMatrix = new ComputationMode(MatrixMatrix); /*!< Computes the kernel function
                                                                                          for all the vectors
                                                                                          in the sets X and Y */
}
