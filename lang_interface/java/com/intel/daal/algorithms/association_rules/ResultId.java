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
 * @ingroup association_rules
 * @{
 */
package com.intel.daal.algorithms.association_rules;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__ASSOCIATION_RULES__RESULTID"></a>
 * @brief Available identifiers of results for the association rules algorithm
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

    private static final int LargeItemsets        = 0;
    private static final int LargeItemsetsSupport = 1;
    private static final int AntecedentItemsets   = 2;
    private static final int ConsequentItemsets   = 3;
    private static final int Confidence           = 4;

    public static final ResultId largeItemsets        = new ResultId(LargeItemsets); /*!< Large itemsets            */
    public static final ResultId largeItemsetsSupport = new ResultId(
            LargeItemsetsSupport);                                                   /*!< Support of large itemsets */
    public static final ResultId antecedentItemsets   = new ResultId(
            AntecedentItemsets);                                                     /*!< Antecedent itemsets       */
    public static final ResultId consequentItemsets   = new ResultId(
            ConsequentItemsets);                                                     /*!< Consequent itemsets       */
    public static final ResultId confidence           = new ResultId(Confidence);    /*!< Confidence                */
}
/** @} */
