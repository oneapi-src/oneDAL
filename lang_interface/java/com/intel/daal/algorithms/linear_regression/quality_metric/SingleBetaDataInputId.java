/* file: SingleBetaDataInputId.java */
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
 * @defgroup linear_regression_quality_metric_single_beta Single Beta Coefficient
 * @ingroup linear_regression_quality_metric_set
 * @{
 */
package com.intel.daal.algorithms.linear_regression.quality_metric;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LINEAR_REGRESSION__QUALITY_METRIC__SINGLEBETADATAINPUTID"></a>
 * @brief Available identifiers of input objects for a single beta quality metrics
 */
public final class SingleBetaDataInputId {
    private int _value;

    /**
     * Constructs the input object identifier using the provided value
     * @param value     Value corresponding to the input object identifier
     */
    public SingleBetaDataInputId(int value) {
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

    /*!< Expected responses (Y), dependent variables */
    public static final SingleBetaDataInputId expectedResponses   = new SingleBetaDataInputId(ExpectedResponses);
    /*!< Predicted responses (Z) */
    public static final SingleBetaDataInputId predictedResponses = new SingleBetaDataInputId(PredictedResponses);
}
/** @} */
