/* file: InitStep2MasterCollectionInputId.java */
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
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__STEP2MASTERCOLLECTIONINPUTID"></a>
 * @brief Available identifiers of input numeric table objects for the DBSCAN algorithm in the eleventh step
 *        of the distributed processing mode
 */
public final class InitStep2MasterCollectionInputId {
    private int _value;

    /**
     * Constructs the input object identifier using the provided value
     * @param value     Value corresponding to the input object identifier
     */
    public InitStep2MasterCollectionInputId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the input object identifier
     * @return Value corresponding to the input object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int step2MeanDependentVariableValue = 0;
    private static final int step2NumberOfRowsValue = 1;
    private static final int step2BinBordersValue = 2;
    private static final int step2BinSizesValue = 3;

    public static final InitStep2MasterCollectionInputId step2MeanDependentVariable = new InitStep2MasterCollectionInputId(step2MeanDependentVariableValue);
        /*!<  */
    public static final InitStep2MasterCollectionInputId step2NumberOfRows = new InitStep2MasterCollectionInputId(step2NumberOfRowsValue);
        /*!<  */
    public static final InitStep2MasterCollectionInputId step2BinBorders = new InitStep2MasterCollectionInputId(step2BinBordersValue);
        /*!<  */
    public static final InitStep2MasterCollectionInputId step2BinSizes = new InitStep2MasterCollectionInputId(step2BinSizesValue);
        /*!<  */
}
/** @} */
