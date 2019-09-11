/* file: InputId.java */
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
 * @defgroup gbt_regression_training Training
 * @brief Contains classes for training the gbt regression model
 * @ingroup gbt_regression
 * @{
 */
/**
 * @brief Contains classes for training the regression model
 */
package com.intel.daal.algorithms.gbt.regression.training;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__REGRESSION__TRAINING__INPUTID"></a>
 * @brief Available identifiers of input objects for the gbt regression algorithm
 */
public final class InputId {
    private int _value;

    /**
     * Constructs the input object identifier using the provided value
     * @param value     Value corresponding to the input object identifier
     */
    public InputId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the input object identifier
     * @return Value corresponding to the input object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int dataId    = 0;
    private static final int dependentVariableId  = 1;

    public static final InputId data = new InputId(dataId);    /*!< Input data table */
    public static final InputId dependentVariable = new InputId(dependentVariableId);  /*!< Values of the dependent variable for the input data */
}
/** @} */
