/* file: TrainingPartialResult.java */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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

package com.intel.daal.algorithms.multinomial_naive_bayes.training;

import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.classifier.training.PartialResultId;
import com.intel.daal.algorithms.multinomial_naive_bayes.PartialModel;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTINOMIAL_NAIVE_BAYES__TRAINING__TRAININGPARTIALRESULT"></a>
 * @brief Provides methods to access results obtained with the compute() method of the
 *        naive Bayes training algorithm
 *        in the online or distributed processing mode
*/

public final class TrainingPartialResult extends com.intel.daal.algorithms.classifier.training.TrainingPartialResult {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    public TrainingPartialResult(DaalContext context) {
        super(context);
        cObject = cNewPartialResult();
    }

    public TrainingPartialResult(DaalContext context, long cObject) {
        super(context);
        this.cObject = cObject;
    }

    /**
     * Returns partial result of the naive Bayes training algorithm
     * @param id   Identifier of the result
     * @return     Result that corresponds to the given identifier
     */
    public PartialModel get(PartialResultId id) {
        if (id == PartialResultId.partialModel) {
            return new PartialModel(getContext(), cGetPartialModel(cObject, PartialResultId.partialModel.getValue()));
        } else {
            return null;
        }
    }

    private native long cNewPartialResult();

    private native long cGetPartialModel(long resAddr, int id);
}
