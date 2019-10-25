/* file: Step4LocalCollectionInputId.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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
