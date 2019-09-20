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
 * @defgroup decision_tree_regression_training Training
 * @brief Contains classes for training the decision_tree regression model
 * @ingroup decision_tree_regression
 * @{
 */
/**
 * @brief Contains classes for training the regression model
 */
package com.intel.daal.algorithms.decision_tree.regression.training;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DECISION_TREE__REGRESSION__TRAINING__TRAININGINPUTID"></a>
 * @brief Available identifiers of the results in the training stage of decision tree
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

    private static final int dataId                          = 0;
    private static final int dependentVariablesId            = 1;
    private static final int dataForPruningId                = 2;
    private static final int dependentVariablesForPruningId  = 3;
    private static final int weightsId                       = 4;

    public static final TrainingInputId data                         = new TrainingInputId(dataId);
        /*!< Input data table */
    public static final TrainingInputId dependentVariables           = new TrainingInputId(dependentVariablesId);
        /*!< Values of the dependent variable for the input data */
    public static final TrainingInputId dataForPruning               = new TrainingInputId(dataForPruningId);
        /*!< Pruning data set */
    public static final TrainingInputId dependentVariablesForPruning = new TrainingInputId(dependentVariablesForPruningId);
        /*!< Labels of the pruning data set */
    public static final TrainingInputId weights                      = new TrainingInputId(weightsId);
        /*!< Optional. Weights of the observations in the training data set */

    public static boolean validate(TrainingInputId id) {
        return id == data ||
               id == dependentVariables ||
               id == dataForPruning ||
               id == dependentVariablesForPruning ||
               id == weights;
    }

    public static void throwIfInvalid(TrainingInputId id) {
        if (id == null) {
            throw new IllegalArgumentException("Null input id");
        }
        if (!TrainingInputId.validate(id)) {
            throw new IllegalArgumentException("Unsupported input id");
        }
    }

}
/** @} */
