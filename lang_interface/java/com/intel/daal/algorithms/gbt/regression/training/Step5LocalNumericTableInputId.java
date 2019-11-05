/* file: Step5LocalNumericTableInputId.java */
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
 * @ingroup gbt_compute
 * @{
 */
package com.intel.daal.algorithms.gbt.regression.training;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__REGRESSION__TRAINING__STEP5LOCALNUMERICTABLEINPUTID"></a>
 * @brief Available identifiers of input data numeric table objects for model-based training in the fifth step
 *        of the distributed processing mode
 */
public final class Step5LocalNumericTableInputId {
    private int _value;

    /**
     * Constructs the input object identifier using the provided value
     * @param value     Value corresponding to the input object identifier
     */
    public Step5LocalNumericTableInputId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the input object identifier
     * @return Value corresponding to the input object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int step5BinnedDataValue = 0;
    private static final int step5TransposedBinnedDataValue = 1;
    private static final int step5BinSizesValue = 2;
    private static final int step5InputTreeStructureValue = 3;
    private static final int step5InputTreeOrderValue = 4;

    public static final Step5LocalNumericTableInputId step5BinnedData = new Step5LocalNumericTableInputId(step5BinnedDataValue);
        /*!<  */
    public static final Step5LocalNumericTableInputId step5TransposedBinnedData = new Step5LocalNumericTableInputId(step5TransposedBinnedDataValue);
        /*!<  */
    public static final Step5LocalNumericTableInputId step5BinSizes = new Step5LocalNumericTableInputId(step5BinSizesValue);
        /*!<  */
    public static final Step5LocalNumericTableInputId step5InputTreeStructure = new Step5LocalNumericTableInputId(step5InputTreeStructureValue);
        /*!<  */
    public static final Step5LocalNumericTableInputId step5InputTreeOrder = new Step5LocalNumericTableInputId(step5InputTreeOrderValue);
        /*!<  */
}
/** @} */
