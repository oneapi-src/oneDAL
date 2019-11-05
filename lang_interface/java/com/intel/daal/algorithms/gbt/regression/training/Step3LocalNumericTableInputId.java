/* file: Step3LocalNumericTableInputId.java */
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
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__REGRESSION__TRAINING__STEP3LOCALNUMERICTABLEINPUTID"></a>
 * @brief Available identifiers of input data numeric table objects for model-based training in the third step
 *        of the distributed processing mode
 */
public final class Step3LocalNumericTableInputId {
    private int _value;

    /**
     * Constructs the input object identifier using the provided value
     * @param value     Value corresponding to the input object identifier
     */
    public Step3LocalNumericTableInputId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the input object identifier
     * @return Value corresponding to the input object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int step3BinnedDataValue = 0;
    private static final int step3BinSizesValue = 1;
    private static final int step3InputTreeStructureValue = 2;
    private static final int step3InputTreeOrderValue = 3;
    private static final int step3OptCoeffsValue = 4;

    public static final Step3LocalNumericTableInputId step3BinnedData = new Step3LocalNumericTableInputId(step3BinnedDataValue);
        /*!<  */
    public static final Step3LocalNumericTableInputId step3BinSizes = new Step3LocalNumericTableInputId(step3BinSizesValue);
        /*!<  */
    public static final Step3LocalNumericTableInputId step3InputTreeStructure = new Step3LocalNumericTableInputId(step3InputTreeStructureValue);
        /*!<  */
    public static final Step3LocalNumericTableInputId step3InputTreeOrder = new Step3LocalNumericTableInputId(step3InputTreeOrderValue);
        /*!<  */
    public static final Step3LocalNumericTableInputId step3OptCoeffs = new Step3LocalNumericTableInputId(step3OptCoeffsValue);
        /*!<  */
}
/** @} */
