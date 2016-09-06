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

package com.intel.daal.algorithms.association_rules;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__ASSOCIATION_RULES__RESULTID"></a>
 * @brief Available identifiers of results for the association rules algorithm
 */
public final class ResultId {
    private int _value;

    public ResultId(int value) {
        _value = value;
    }

    /**
     * Returns the result identifier
     * @return Value of the result identifier
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
