/* file: DistributedPartialResultStep1Id.java */
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
 * @ingroup gbt_distributed
 * @{
 */
package com.intel.daal.algorithms.gbt.regression.training;

import com.intel.daal.utils.*;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__REGRESSION__TRAINING__DISTRIBUTEDPARTIALRESULTSTEP1ID"></a>
 * @brief Available identifiers of partial results of the model-based training in the first step
 *        of the distributed processing mode
 */
public final class DistributedPartialResultStep1Id {
    private int _value;

    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the partial result object identifier using the provided value
     * @param value     Value corresponding to the partial result object identifier
     */
    public DistributedPartialResultStep1Id(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the partial result object identifier
     * @return Value corresponding to the partial result object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int responseValue = 0;
    private static final int optCoeffsValue = 1;
    private static final int treeOrderValue = 2;
    private static final int finalizedTreeValue = 3;
    private static final int step1TreeStructureValue = 4;

    public static final DistributedPartialResultStep1Id response = new DistributedPartialResultStep1Id(responseValue);
        /*!<  */
    public static final DistributedPartialResultStep1Id optCoeffs = new DistributedPartialResultStep1Id(optCoeffsValue);
        /*!<  */
    public static final DistributedPartialResultStep1Id treeOrder = new DistributedPartialResultStep1Id(treeOrderValue);
        /*!<  */
    public static final DistributedPartialResultStep1Id finalizedTree = new DistributedPartialResultStep1Id(finalizedTreeValue);
        /*!<  */
    public static final DistributedPartialResultStep1Id step1TreeStructure = new DistributedPartialResultStep1Id(step1TreeStructureValue);
        /*!<  */
}
/** @} */
