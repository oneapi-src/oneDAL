/* file: BackwardResultId.java */
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

package com.intel.daal.algorithms.neural_networks.layers;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__BACKWARDRESULTID"></a>
 * \brief Available identifiers of results for the backward layer
 */
public final class BackwardResultId {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    private int _value;

    /**
     * Constructs the result object identifier using the provided value
     * @param value     Value of the input object identifier
     */
    public BackwardResultId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the result identifier
     * @return Value corresponding to the result identifier
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
