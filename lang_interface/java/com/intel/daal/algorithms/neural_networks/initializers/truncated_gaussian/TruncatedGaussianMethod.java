/* file: TruncatedGaussianMethod.java */
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
 * @defgroup initializers_truncated_gaussian Truncated Gaussian Initializer
 * @brief Contains classes for neural network weights and biases truncated gaussian initializer
 * @ingroup initializers
 * @{
 */
package com.intel.daal.algorithms.neural_networks.initializers.truncated_gaussian;

import com.intel.daal.utils.*;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__INITIALIZERS__TRUNCATED_GAUSSIAN__TRUNCATEDGAUSSIANMETHOD"></a>
 * @brief Available methods for the truncated gaussian initializer
 */
public final class TruncatedGaussianMethod {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    private int _value;

    /**
     * Constructs the method object using the provided value
     * @param value     Value corresponding to the method object
     */
    public TruncatedGaussianMethod(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the method object
     * @return Value corresponding to the method object
     */
    public int getValue() {
        return _value;
    }

    private static final int defaultDenseId = 0;

    public static final TruncatedGaussianMethod defaultDense = new TruncatedGaussianMethod(defaultDenseId); /*!< Default: performance-oriented method */
}
/** @} */
