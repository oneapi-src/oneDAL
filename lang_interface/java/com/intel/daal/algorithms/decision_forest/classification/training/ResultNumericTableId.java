/* file: ResultNumericTableId.java */
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
 * @ingroup decision_forest_classification_training
 * @{
 */
package com.intel.daal.algorithms.decision_forest.classification.training;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DECISION_FOREST__CLASSIFICATION__TRAINING_RESULTNUMERICTABLEID"></a>
 * @brief Available identifiers of the result of decision forest model-based training
 */
public final class ResultNumericTableId {
    private int _value;

    /**
     * Constructs the result numeric table object identifier using the provided value
     * @param value     Value corresponding to the result numeric table object identifier
     */
    public ResultNumericTableId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the result numeric table object identifier
     * @return Value corresponding to the result numeric table object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int outOfBagErrorId = 1;
    private static final int variableImportanceId = 2;
    private static final int outOfBagErrorPerObservationId = 3;

    public static final ResultNumericTableId outOfBagError = new ResultNumericTableId(outOfBagErrorId);
                        /*!< %Out-of-bag error result */
    public static final ResultNumericTableId variableImportance = new ResultNumericTableId(variableImportanceId);
                        /*!< %Variable importance result */
    public static final ResultNumericTableId outOfBagErrorPerObservation = new ResultNumericTableId(outOfBagErrorPerObservationId);
                        /*!< %Out-of-bag error perf observation result */
}
/** @} */
