/* file: TrainingInputId.java */
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
 * @ingroup decision_tree_classification_training
 * @{
 */
package com.intel.daal.algorithms.decision_tree.classification.training;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DECISION_TREE__CLASSIFICATION__TRAINING_TRAININGINPUTID"></a>
 * @brief Available identifiers of the result of decision tree model-based training
 */
public final class TrainingInputId {
    private int _value;

    /**
     * Constructs the training input object identifier using the provided value
     * @param value     Value corresponding to the training input object identifier
     */
    public TrainingInputId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the training input object identifier
     * @return Value corresponding to the training input object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int dataForPruningId   = 3;
    private static final int labelsForPruningId = 4;

    public static final TrainingInputId dataForPruning   = new TrainingInputId(dataForPruningId);   /*!< Pruning data set */
    public static final TrainingInputId labelsForPruning = new TrainingInputId(labelsForPruningId); /*!< Labels of the pruning data set */
}
/** @} */
