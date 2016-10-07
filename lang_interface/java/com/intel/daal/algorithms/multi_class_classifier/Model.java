/* file: Model.java */
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

package com.intel.daal.algorithms.multi_class_classifier;

import com.intel.daal.services.DaalContext;
import com.intel.daal.data_management.data.Factory;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTI_CLASS_CLASSIFIER__MODEL"></a>
 * @brief Model of the classifier trained by the multi_class_classifier.training.TrainingBatch algorithm.
 */
public class Model extends com.intel.daal.algorithms.classifier.Model {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    public Model(DaalContext context, long cModel) {
        super(context, cModel);
    }

    /**
     * Returns a two-class classifier model in a multi-class classifier model
     * @param idx   Index of two-class classifier model in a collection
     * @return      Two-class classifier model
     */
    public com.intel.daal.algorithms.classifier.Model getTwoClassClassifierModel(long idx) {
        return (com.intel.daal.algorithms.classifier.Model)Factory.instance().createObject(getContext(), cGetTwoClassClassifierModel(this.getCObject(), idx));
    }

    /**
     * Returns a number of two-class classifiers associated with the model
     * @return Number of two-class classifiers associated with the model
     */
    public long getNumberOfTwoClassClassifierModels() {
        return cGetNumberOfTwoClassClassifierModels(this.getCObject());
    }

    private native long cGetNumberOfTwoClassClassifierModels(long modelAddr);
    private native long cGetTwoClassClassifierModel(long modelAddr, long idx);
}
