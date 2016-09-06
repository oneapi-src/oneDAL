/* file: GroupOfBetasResultId.java */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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

package com.intel.daal.algorithms.linear_regression.quality_metric;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LINEAR_REGRESSION__QUALITY_METRIC__GROUPOFBETASRESULTID"></a>
 * @brief Available identifiers of the result of single beta quality metrics
 */
public final class GroupOfBetasResultId {
    private int _value;

    public GroupOfBetasResultId(int value) {
        _value = value;
    }

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
