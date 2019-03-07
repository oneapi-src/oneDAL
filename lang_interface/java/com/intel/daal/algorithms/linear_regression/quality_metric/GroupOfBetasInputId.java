/* file: GroupOfBetasInputId.java */
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
 * @ingroup linear_regression_quality_metric_group_of_betas
 * @{
 */
package com.intel.daal.algorithms.linear_regression.quality_metric;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LINEAR_REGRESSION__QUALITY_METRIC__GROUPOFBETASINPUTID"></a>
 * @brief Available identifiers of input objects for a single beta quality metrics
 */
public final class GroupOfBetasInputId {
    private int _value;

    /**
     * Constructs the input object identifier using the provided value
     * @param value     Value corresponding to the input object identifier
     */
    public GroupOfBetasInputId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the input object identifier
     * @return Value corresponding to the input object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int ExpectedResponses   = 0;
    private static final int PredictedResponses = 1;
    private static final int PredictedReducedModelResponses = 2;

    /*!< Expected responses (Y), dependent variables */
    public static final GroupOfBetasInputId expectedResponses   = new GroupOfBetasInputId(ExpectedResponses);
    /*!< Predicted responses (Z) */
    public static final GroupOfBetasInputId predictedResponses = new GroupOfBetasInputId(PredictedResponses);
    /*!< Responses predicted by reduced model where p - p0 of p betas are set to zero */
    public static final GroupOfBetasInputId predictedReducedModelResponses = new GroupOfBetasInputId(PredictedReducedModelResponses);
}
/** @} */
