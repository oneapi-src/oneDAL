/* file: GroupOfBetasResultId.java */
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
 * <a name="DAAL-CLASS-ALGORITHMS__LINEAR_REGRESSION__QUALITY_METRIC__GROUPOFBETASRESULTID"></a>
 * @brief Available identifiers of the result of single beta quality metrics
 */
public final class GroupOfBetasResultId {
    private int _value;

    /**
     * Constructs the result object identifier using the provided value
     * @param value     Value corresponding to the result object identifier
     */
    public GroupOfBetasResultId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the result object identifier
     * @return Value corresponding to the result object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int ExpectedMeans   = 0;
    private static final int ExpectedVariance = 1;
    private static final int RegSS = 2;
    private static final int ResSS = 3;
    private static final int TSS = 4;
    private static final int DeterminationCoeff = 5;
    private static final int FStatistics = 6;

    /*!< Means of expected responses computed for each dependent variable */
    public static final GroupOfBetasResultId expectedMeans   = new GroupOfBetasResultId(ExpectedMeans);
    /*!< Variance of expected responses computed for each dependent variable */
    public static final GroupOfBetasResultId expectedVariance = new GroupOfBetasResultId(ExpectedVariance);
    /*!< Regression sum of squares computed for each dependent variable */
    public static final GroupOfBetasResultId regSS = new GroupOfBetasResultId(RegSS);
    /*!< Sum of squares of residuals computed for each dependent variable */
    public static final GroupOfBetasResultId resSS = new GroupOfBetasResultId(ResSS);
    /*!< Total sum of squares of residuals computed for each dependent variable */
    public static final GroupOfBetasResultId tSS = new GroupOfBetasResultId(TSS);
    /*!< Determination coefficient computed for each dependent variable */
    public static final GroupOfBetasResultId determinationCoeff = new GroupOfBetasResultId(DeterminationCoeff);
    /*!< F-statistics computed for each dependent variable */
    public static final GroupOfBetasResultId fStatistics = new GroupOfBetasResultId(FStatistics);
}
/** @} */
