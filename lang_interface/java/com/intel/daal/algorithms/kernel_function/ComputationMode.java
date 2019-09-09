/* file: ComputationMode.java */
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
 * @ingroup kernel_function
 * @{
 */
package com.intel.daal.algorithms.kernel_function;

import java.lang.annotation.Native;

import com.intel.daal.utils.*;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__KERNEL_FUNCTION__COMPUTATIONMODE"></a>
 * @brief Available modes of kernel function computation
 */
public final class ComputationMode {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    private int _value;

    /**
     * Constructs the computation mode object using the provided value
     * @param value     Value corresponding to the computation mode object
     */
    public ComputationMode(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the computation mode object
     * @return Value corresponding to the computation mode object
     */
    public int getValue() {
        return _value;
    }

    @Native private static final int VectorVector = 0;
    @Native private static final int MatrixVector = 1;
    @Native private static final int MatrixMatrix = 2;

    public static final ComputationMode vectorVector = new ComputationMode(VectorVector); /*!< Computes the kernel function for given feature vectors
                                                                                          Xi and Yj */
    public static final ComputationMode matrixVector = new ComputationMode(MatrixVector); /*!< Computes the kernel function
                                                                                          for all the vectors in the set X and
                                                                                          a given feature vector Yi */
    public static final ComputationMode matrixMatrix = new ComputationMode(MatrixMatrix); /*!< Computes the kernel function
                                                                                          for all the vectors
                                                                                          in the sets X and Y */
}
/** @} */
