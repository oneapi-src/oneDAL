/* file: BatchNormalizationLayerDataId.java */
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
 * @defgroup batch_normalization Batch Normalization Layer
 * @brief Contains classes for batch normalization layer
 * @ingroup layers
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.batch_normalization;

import com.intel.daal.utils.*;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__BATCHNORMALIZATIONBATCH_NORMALIZATION__BATCHNORMALIZATIONLAYERDATAID"></a>
 * \brief Identifiers of input objects for the backward batch normalization layer and results for the forward batch normalization layer
 */
public final class BatchNormalizationLayerDataId {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    private int _value;

    /**
     * Constructs the input object identifier using the provided value
     * @param value     Value corresponding to the input object identifier
     */
    public BatchNormalizationLayerDataId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the input object identifier
     * @return Value corresponding to the input object identifier
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
/** @} */
