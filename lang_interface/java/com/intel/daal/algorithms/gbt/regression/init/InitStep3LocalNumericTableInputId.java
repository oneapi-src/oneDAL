/* file: InitStep3LocalNumericTableInputId.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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

/**
 * @ingroup dbscan_compute
 * @{
 */
package com.intel.daal.algorithms.gbt.regression.init;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__STEp31LOCALNUMERICTABLEINPUTID"></a>
 * @brief Available identifiers of input numeric table objects for the DBSCAN algorithm in the eleventh step
 *        of the distributed processing mode
 */
public final class InitStep3LocalNumericTableInputId {
    private int _value;

    /**
     * Constructs the input object identifier using the provided value
     * @param value     Value corresponding to the input object identifier
     */
    public InitStep3LocalNumericTableInputId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the input object identifier
     * @return Value corresponding to the input object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int step3MergedBinBordersValue = 0;
    private static final int step3BinQuantitiesValue = 1;
    private static final int step3LocalDataValue = 2;
    private static final int step3InitialResponseValue = 3;

    public static final InitStep3LocalNumericTableInputId step3MergedBinBorders = new InitStep3LocalNumericTableInputId(step3MergedBinBordersValue);
        /*!<  */
    public static final InitStep3LocalNumericTableInputId step3BinQuantities = new InitStep3LocalNumericTableInputId(step3BinQuantitiesValue);
       /*!<  */
    public static final InitStep3LocalNumericTableInputId step3LocalData = new InitStep3LocalNumericTableInputId(step3LocalDataValue);
    /*!<  */
    public static final InitStep3LocalNumericTableInputId step3InitialResponse = new InitStep3LocalNumericTableInputId(step3InitialResponseValue);
    /*!<  */
}
/** @} */
