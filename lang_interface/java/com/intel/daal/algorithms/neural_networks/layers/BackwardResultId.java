/* file: BackwardResultId.java */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
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
 * @ingroup layers_backward
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers;

import com.intel.daal.utils.*;
import com.intel.daal.utils.*;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__BACKWARDRESULTID"></a>
 * \brief Available identifiers of results for the backward layer
 */
public final class BackwardResultId {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    private int _value;

    /**
     * Constructs the result object identifier using the provided value
     * @param value     Value corresponding to the result object identifier
     */
    public BackwardResultId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the result object identifier
     * @return Value corresponding to the result object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int gradientId = 0;
    private static final int weightDerivativesId = 1;
    private static final int biasDerivativesId = 2;

    public static final BackwardResultId gradient = new BackwardResultId(gradientId); /*!< Gradient with respect to the outputs */
    public static final BackwardResultId weightDerivatives = new BackwardResultId(weightDerivativesId); /*!< Gradient with respect to the weight */
    public static final BackwardResultId biasDerivatives = new BackwardResultId(biasDerivativesId); /*!< Gradient with respect to the bias */
}
/** @} */
