/* file: DistributedPartialResultStep3Id.java */
/*******************************************************************************
* Copyright 2014-2022 Intel Corporation
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
 * @ingroup svd_distributed
 * @{
 */
package com.intel.daal.algorithms.svd;

import com.intel.daal.utils.*;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__SVD__DISTRIBUTEDPARTIALRESULTSTEP3ID"></a>
 * @brief Available types of partial results obtained in the second step of the SVD algorithm in the distributed processing mode, stored in
 * the Result object
 */
public final class DistributedPartialResultStep3Id {
    private int _value;

    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the partial result object identifier using the provided value
     * @param value     Value corresponding to the partial result object identifier
     */
    public DistributedPartialResultStep3Id(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the partial result object identifier
     * @return Value corresponding to the partial result object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int finalResultFromStep3Id = 0;

    /** Result object with singular values and the right orthogonal matrix */
    public static final DistributedPartialResultStep3Id finalResultFromStep3 = new DistributedPartialResultStep3Id(
            finalResultFromStep3Id);
}
/** @} */
