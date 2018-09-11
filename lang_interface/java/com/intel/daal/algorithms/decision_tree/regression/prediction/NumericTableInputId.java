/* file: NumericTableInputId.java */
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
 * @ingroup decision_tree_regression_prediction
 * @{
 */
package com.intel.daal.algorithms.decision_tree.regression.prediction;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DECISION_TREE__REGRESSION__PREDICTION__NUMERICTABLEINPUTID"></a>
 * @brief Available identifiers of input numeric tables for making decision tree model-based prediction
 */
public final class NumericTableInputId {
    private int _value;

    /**
     * Constructs the input numeric tables object identifier using the provided value
     * @param value     Value corresponding to the input numeric tables object identifier
     */
    public NumericTableInputId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the input numeric tables object identifier
     * @return Value corresponding to the input numeric tables object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int dataId = 0;

    public static final NumericTableInputId data = new NumericTableInputId(dataId); /*!< Input data table */
}
/** @} */
