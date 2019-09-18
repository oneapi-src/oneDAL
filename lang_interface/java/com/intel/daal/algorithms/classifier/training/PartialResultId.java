/* file: PartialResultId.java */
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
 * @ingroup training
 * @{
 */
package com.intel.daal.algorithms.classifier.training;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__CLASSIFIER__TRAINING__PARTIALRESULTID"></a>
 * @brief Available identifiers of partial results of the classification algorithm
 */
public final class PartialResultId {
    private int _value;

    /**
     * Constructs the partial result object identifier using the provided value
     * @param value     Value corresponding to the partial result object identifier
     */
    public PartialResultId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the partial result object identifier
     * @return Value corresponding to the partial result object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int PartialModel = 0;

    /** Trained model */
    public static final PartialResultId partialModel = new PartialResultId(PartialModel);

    public static boolean validate(PartialResultId id) {
        return id.getValue() == partialModel.getValue();
    }

    public static void throwIfInvalid(PartialResultId id) {
        if (id == null) {
            throw new IllegalArgumentException("Null result id");
        }
        if (!PartialResultId.validate(id)) {
            throw new IllegalArgumentException("Unsupported result id");
        }
    }
}
/** @} */
