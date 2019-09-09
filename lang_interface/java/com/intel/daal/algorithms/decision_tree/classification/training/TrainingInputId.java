/* file: TrainingInputId.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
