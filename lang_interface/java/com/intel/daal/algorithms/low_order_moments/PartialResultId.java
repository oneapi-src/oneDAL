/* file: PartialResultId.java */
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
 * <a name="DAAL-CLASS-ALGORITHMS__LOW_ORDER_MOMENTS__PARTIALRESULTID"></a>
 * @brief Available identifiers of partial results of the low order %moments algorithm
 */
public final class PartialResultId {
    private int _value;

    public PartialResultId(int value) {
        _value = value;
    }

    public int getValue() {
        return _value;
    }

    private static final int NObservations             = 0;
    private static final int PartialMinimum            = 1;
    private static final int PartialMaximum            = 2;
    private static final int PartialSum                = 3;
    private static final int PartialSumSquares         = 4;
    private static final int PartialSumSquaresCentered = 5;

    /**< Number of rows processed so far */
    public static final PartialResultId nObservations = new PartialResultId(NObservations);

    /**< Partial minimum */
    public static final PartialResultId partialMinimum = new PartialResultId(PartialMinimum);

    /*!< Partial maximum */
    public static final PartialResultId partialMaximum = new PartialResultId(PartialMaximum);

    /*!< Partial sum */
    public static final PartialResultId partialSum = new PartialResultId(PartialSum);

    /*!< Partial sum of squares */
    public static final PartialResultId partialSumSquares = new PartialResultId(PartialSumSquares);

    /**< Partial sum of squared difference from the means */
    public static final PartialResultId partialSumSquaresCentered = new PartialResultId(PartialSumSquaresCentered);
}
