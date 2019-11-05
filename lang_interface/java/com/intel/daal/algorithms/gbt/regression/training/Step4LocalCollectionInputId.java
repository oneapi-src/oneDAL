/* file: Step4LocalCollectionInputId.java */
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
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__REGRESSION__TRAINING__STEP4LOCALCOLLECTIONINPUTID"></a>
 * @brief Available identifiers of input data collection table objects for model-based training in the fourth step
 *        of the distributed processing mode
 */
public final class Step4LocalCollectionInputId {
    private int _value;

    /**
     * Constructs the input object identifier using the provided value
     * @param value     Value corresponding to the input object identifier
     */
    public Step4LocalCollectionInputId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the input object identifier
     * @return Value corresponding to the input object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int step4FeatureIndicesValue = 1;
    private static final int step4ParentTotalHistogramsValue = 2;
    private static final int step4PartialHistogramsValue = 3;

    public static final Step4LocalCollectionInputId step4FeatureIndices = new Step4LocalCollectionInputId(step4FeatureIndicesValue);
        /*!<  */
    public static final Step4LocalCollectionInputId step4ParentTotalHistograms = new Step4LocalCollectionInputId(step4ParentTotalHistogramsValue);
        /*!<  */
    public static final Step4LocalCollectionInputId step4PartialHistograms = new Step4LocalCollectionInputId(step4PartialHistogramsValue);
        /*!<  */
}
/** @} */
