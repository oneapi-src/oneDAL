/* file: ResultId.java */
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
 * @ingroup low_order_moments
 * @{
 */
package com.intel.daal.algorithms.low_order_moments;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LOW_ORDER_MOMENTS__RESULTID"></a>
 * @brief Available types of results of the low order %moments algorithm
 */
public final class ResultId {
    private int _value;

    /**
     * Constructs the result object identifier using the provided value
     * @param value     Value corresponding to the result object identifier
     */
    public ResultId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the result object identifier
     * @return Value corresponding to the result object identifier
     */
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
/** @} */
