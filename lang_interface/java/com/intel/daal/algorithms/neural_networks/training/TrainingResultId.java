/* file: TrainingResultId.java */
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
 * @ingroup neural_networks_training
 * @{
 */
package com.intel.daal.algorithms.neural_networks.training;

import com.intel.daal.utils.*;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__TRAINING__TRAININGRESULTID"></a>
 * \brief Available identifiers of result of the neural network model based training
 */
public final class TrainingResultId {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    private int _value;

    /**
     * Constructs the training result object identifier using the provided value
     * @param value     Value corresponding to the training result object identifier
     */
    public TrainingResultId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the training result object identifier
     * @return Value corresponding to the training result object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int modelId = 0;

    public static final TrainingResultId model = new TrainingResultId(modelId); /*!< Neural network Model */
}
/** @} */
