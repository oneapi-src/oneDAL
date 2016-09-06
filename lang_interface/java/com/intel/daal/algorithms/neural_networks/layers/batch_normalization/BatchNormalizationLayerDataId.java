/* file: BatchNormalizationLayerDataId.java */
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

package com.intel.daal.algorithms.neural_networks.layers.batch_normalization;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__BATCHNORMALIZATIONBATCH_NORMALIZATION__BATCHNORMALIZATIONLAYERDATAID"></a>
 * \brief Identifiers of input objects for the backward batch normalization layer and results for the forward batch normalization layer
 */
public final class BatchNormalizationLayerDataId {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    private int _value;

    /**
     * Constructs the result object identifier using the provided value
     * @param value     Value of the input object identifier
     */
    public BatchNormalizationLayerDataId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the result identifier
     * @return Value corresponding to the result identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int auxDataId = 0;
    private static final int auxWeightsId = 1;
    private static final int auxMeanId = 2;
    private static final int auxStandardDeviationId = 3;
    private static final int auxPopulationMeanId = 4;
    private static final int auxPopulationVarianceId = 5;

    public static final BatchNormalizationLayerDataId auxData               = new BatchNormalizationLayerDataId(auxDataId);
            /*!< p-dimensional tensor that stores forward batch normalization layer input data */
    public static final BatchNormalizationLayerDataId auxWeights            = new BatchNormalizationLayerDataId(auxWeightsId);
            /*!< 1-dimensional tensor of size \f$n_k\f$ that stores input weights for forward batch normalization layer */
    public static final BatchNormalizationLayerDataId auxMean               = new BatchNormalizationLayerDataId(auxMeanId);
            /*!< 1-dimensional tensor of size \f$n_k\f$ that stores mini-batch mean */
    public static final BatchNormalizationLayerDataId auxStandardDeviation  = new BatchNormalizationLayerDataId(auxStandardDeviationId);
            /*!< 1-dimensional tensor of size \f$n_k\f$ that stores mini-batch standard deviation */
    public static final BatchNormalizationLayerDataId auxPopulationMean     = new BatchNormalizationLayerDataId(auxPopulationMeanId);
            /*!< 1-dimensional tensor of size \f$n_k\f$ that stores resulting population mean */
    public static final BatchNormalizationLayerDataId auxPopulationVariance = new BatchNormalizationLayerDataId(auxPopulationVarianceId);
            /*!< 1-dimensional tensor of size \f$n_k\f$ that stores resulting population variance */
}
