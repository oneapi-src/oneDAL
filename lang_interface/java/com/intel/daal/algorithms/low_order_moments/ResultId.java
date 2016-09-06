/* file: ResultId.java */
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

package com.intel.daal.algorithms.low_order_moments;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LOW_ORDER_MOMENTS__RESULTID"></a>
 * @brief Available types of results of the low order %moments algorithm
 */
public final class ResultId {
    private int _value;

    public ResultId(int value) {
        _value = value;
    }

    public int getValue() {
        return _value;
    }

    private static final int Minimum              = 0;
    private static final int Maximum              = 1;
    private static final int Sum                  = 2;
    private static final int SumSquares           = 3;
    private static final int SumSquaresCentered   = 4;
    private static final int Mean                 = 5;
    private static final int SecondOrderRawMoment = 6;
    private static final int Variance             = 7;
    private static final int StandardDeviation    = 8;
    private static final int Variation            = 9;

    public static final ResultId minimum              = new ResultId(Minimum);           /*!< Minimum */
    public static final ResultId maximum              = new ResultId(Maximum);           /*!< Maximum */
    public static final ResultId sum                  = new ResultId(Sum);               /*!< Sum */
    public static final ResultId sumSquares           = new ResultId(SumSquares);        /*!< Sum of squares */
    public static final ResultId sumSquaresCentered   = new ResultId(
            SumSquaresCentered);                                                         /*!< Sum of squared difference from the means */
    public static final ResultId mean                 = new ResultId(Mean);              /*!< Mean */
    public static final ResultId secondOrderRawMoment = new ResultId(
            SecondOrderRawMoment);                                                       /*!< Second raw order moment */
    public static final ResultId variance             = new ResultId(Variance);          /*!< Variance */
    public static final ResultId standardDeviation    = new ResultId(StandardDeviation); /*!< Standard deviation */
    public static final ResultId variation            = new ResultId(Variation);         /*!< Variation */
}
