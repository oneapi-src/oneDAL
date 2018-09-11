/* file: TrainingDistributedInputId.java */
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
 * @ingroup multinomial_naive_bayes_training_distributed
 * @{
 */
package com.intel.daal.algorithms.multinomial_naive_bayes.training;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTINOMIAL_NAIVE_BAYES__TRAINING__TRAININGDISTRIBUTEDINPUTID"></a>
 * @brief Available identifiers of input objects of the classifier model training algorithm
 */
public final class TrainingDistributedInputId {
    private int _value;

    /**
     * Constructs the training input object identifier using the provided value
     * @param value     Value corresponding to the training input object identifier
     */
    public TrainingDistributedInputId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the training input object identifier
     * @return Value corresponding to the training input object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int PartialModels = 0;

    public static final TrainingDistributedInputId partialModels = new TrainingDistributedInputId(
            PartialModels); /*!< Data for the training stage */
}
/** @} */
