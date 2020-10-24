/* file: DataUseInModelId.java */
/*******************************************************************************
* Copyright 2014-2020 Intel Corporation
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
 * @ingroup bf_knn_classification
 * @{
 */
package com.intel.daal.algorithms.bf_knn_classification;

import com.intel.daal.utils.*;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__BF_KNN_CLASSIFICATION__DATAUSEINMODELID"></a>
 * @brief The option to enable/disable an usage of the input dataset in k nearest neighbors model
 */
public final class DataUseInModelId {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    private int _value;

    /**
     * Constructs the input data usage mode object identifier using the provided value
     * @param value     Value corresponding to the input data usage mode object identifier
     */
    public DataUseInModelId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the input data usage mode object identifier
     * @return Value corresponding to the input data usage mode object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int doNotUseId = 0;
    private static final int doUseId    = 1;

    public static final DataUseInModelId doNotUse = new DataUseInModelId(doNotUseId);
        /*!< The input data and labels will not be the component of the trained kNN model */
    public static final DataUseInModelId doUse    = new DataUseInModelId(doUseId);
        /*!< The input data and labels will be the component of the trained kNN model */
}
/** @} */
