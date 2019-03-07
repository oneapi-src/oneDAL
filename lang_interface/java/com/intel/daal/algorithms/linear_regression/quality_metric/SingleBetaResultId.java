/* file: SingleBetaResultId.java */
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
 * @ingroup linear_regression_quality_metric_single_beta
 * @{
 */
package com.intel.daal.algorithms.linear_regression.quality_metric;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LINEAR_REGRESSION__QUALITY_METRIC__SINGLEBETARESULTID"></a>
 * @brief Available identifiers of the result of single beta quality metrics
 */
public final class SingleBetaResultId {
    private int _value;

    /**
     * Constructs the result object identifier using the provided value
     * @param value     Value corresponding to the result object identifier
     */
    public SingleBetaResultId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the result object identifier
     * @return Value corresponding to the result object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int Rms   = 0;
    private static final int Variance = 1;
    private static final int ZScore = 2;
    private static final int ConfidenceIntervals = 3;
    private static final int InverseOfXtX = 4;

    /*!< Root means square errors computed for each response (dependent variable) */
    public static final SingleBetaResultId rms   = new SingleBetaResultId(Rms);
    /*!< Variance computed for each response (dependent variable) */
    public static final SingleBetaResultId variance = new SingleBetaResultId(Variance);
    /*!< Z-score statistics used in testing of insignificance one beta coefficient. H0: beta[i]=0 */
    public static final SingleBetaResultId zScore = new SingleBetaResultId(ZScore);
    /*!< Limits of the confidence intervals for each beta */
    public static final SingleBetaResultId confidenceIntervals = new SingleBetaResultId(ConfidenceIntervals);
    /*!< Inverse(Xt * X) matrix */
    public static final SingleBetaResultId inverseOfXtX = new SingleBetaResultId(InverseOfXtX);
}
/** @} */
