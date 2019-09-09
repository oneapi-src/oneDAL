/* file: TrainingDistributedInputId.java */
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
 * @ingroup multinomial_naive_bayes_training_distributed
 * @{
 */
package com.intel.daal.algorithms.multinomial_naive_bayes.training;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTINOMIAL_NAIVE_BAYES__TRAINING__TRAININGDISTRIBUTEDINPUTID"></a>
 * @brief Available identifiers of input objects of the classifier model training algorithm
 */
public final class TrainingDistributedInputId {
    private int _value;

    /**
     * Constructs the training input object identifier using the provided value
     * @param value     Value corresponding to the training input object identifier
     */
    public TrainingDistributedInputId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the training input object identifier
     * @return Value corresponding to the training input object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int PartialModels = 0;

    public static final TrainingDistributedInputId partialModels = new TrainingDistributedInputId(
            PartialModels); /*!< Data for the training stage */
}
/** @} */
