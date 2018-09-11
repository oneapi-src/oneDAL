/* file: BatchNormalizationForwardInputLayerDataId.java */
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
 * @ingroup batch_normalization_forward
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.batch_normalization;

import com.intel.daal.utils.*;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__BATCHNORMALIZATIONBATCH_NORMALIZATION__BATCHNORMALIZATIONFORWARDINPUTLAYERDATAID"></a>
 * \brief Available identifiers of input objects for the forward batch normalization layer
 */
public final class BatchNormalizationForwardInputLayerDataId {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    private int _value;

    /**
     * Constructs the input object identifier using the provided value
     * @param value     Value corresponding to the input object identifier
     */
    public BatchNormalizationForwardInputLayerDataId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the input object identifier
     * @return Value corresponding to the input object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int populationMeanId = 3;
    private static final int populationVarianceId = 4;

    public static final BatchNormalizationForwardInputLayerDataId populationMean     = new BatchNormalizationForwardInputLayerDataId(populationMeanId);
            /*!< 1-dimensional tensor of size \f$n_k\f$ that stores population mean computed on the previous stage */
    public static final BatchNormalizationForwardInputLayerDataId populationVariance = new BatchNormalizationForwardInputLayerDataId(populationVarianceId);
            /*!< 1-dimensional tensor of size \f$n_k\f$ that stores population variance computed on the previous stage */
}
/** @} */
