/* file: ComputationMode.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/**
 * @ingroup kernel_function
 * @{
 */
package com.intel.daal.algorithms.kernel_function;

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
/** @} */
