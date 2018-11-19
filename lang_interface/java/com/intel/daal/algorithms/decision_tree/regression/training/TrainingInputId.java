/* file: TrainingInputId.java */
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

    public static final TrainingInputId data                         = new TrainingInputId(dataId);
        /*!< Input data table */
    public static final TrainingInputId dependentVariables           = new TrainingInputId(dependentVariablesId);
        /*!< Values of the dependent variable for the input data */
    public static final TrainingInputId dataForPruning               = new TrainingInputId(dataForPruningId);
        /*!< Pruning data set */
    public static final TrainingInputId dependentVariablesForPruning = new TrainingInputId(dependentVariablesForPruningId);
        /*!< Labels of the pruning data set */
}
/** @} */
