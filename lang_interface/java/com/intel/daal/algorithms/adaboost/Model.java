/* file: Model.java */
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
 * @ingroup adaboost
 */
package com.intel.daal.algorithms.adaboost;

import com.intel.daal.utils.*;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__ADABOOST__MODEL"></a>
 * @brief %Model of the classifier trained by the AdaBoost algorithm in the batch processing mode.
 */
public class Model extends com.intel.daal.algorithms.classifier.Model {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public Model(DaalContext context, long cModel) {
        super(context, cModel);
    }

    /**
     *  Returns the number of weak learners constructed
     *  during training of the AdaBoost algorithm.
     *  The size of the array equals the number of weak learners
     *  @return The number of weak learners
     */
    public long getNumberOfWeakLearners() {
        return cGetNumberOfWeakLearners(this.cObject);
    }

    /**
     *  Returns weak learner model constructed during training of the AdaBoost algorithm
     *  @param idx  Index of the model in the collection
     *  @return Weak Learner model corresponding to the index idx
     */
    public com.intel.daal.algorithms.classifier.Model getWeakLearnerModel(long idx) {
        return new com.intel.daal.algorithms.classifier.Model(getContext(),
                                                              cGetWeakLearnerModel(this.cObject, idx));
    }

    /**
     *  Add weak learner model into the AdaBoost model
     *  @param model Weak learner model to add into collection
     */
    public void addWeakLearnerModel(com.intel.daal.algorithms.classifier.Model model) {
        cAddWeakLearnerModel(this.cObject, model.getCObject());
    }

    /**
     *  Clears the collecion of weak learners
     */
    public void clearWeakLearnerModels() {
        cClearWeakLearnerModels(this.cObject);
    }

    /**
     *  Retrieves the number of features in the dataset was used on the training stage
     *  @return Number of features in the dataset was used on the training stage
     */
    public long getNumberOfFeatures() {
        return cGetNumberOfFeatures(this.cObject);
    }

    /**
     * Returns the numeric table that contains the array of weights of weak learners constructed
     * during training of the AdaBoost  algorithm.
     * The size of the array equals the number of weak learners
     * @return Array of weights of weak learners.
     */
    public NumericTable getAlpha() {
        return (NumericTable)Factory.instance().createObject(getContext(), cGetAlpha(this.getCObject()));
    }

    private native long cGetNumberOfWeakLearners(long selfPtr);
    private native long cGetWeakLearnerModel(long selfPtr, long idx);
    private native void cAddWeakLearnerModel(long selfPtr, long modelPtr);
    private native void cClearWeakLearnerModels(long selfPtr);
    private native long cGetNumberOfFeatures(long selfPtr);
    private native long cGetAlpha(long selfPtr);
}
/** @} */
