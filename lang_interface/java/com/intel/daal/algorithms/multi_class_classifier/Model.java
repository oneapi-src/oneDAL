/* file: Model.java */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
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
 * @defgroup multi_class_classifier Multi-class Classifier
 * @brief Contains classes for computing the results of the multi-class classifier algorithm
 * @ingroup classification
 * @{
 */
package com.intel.daal.algorithms.multi_class_classifier;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;
import com.intel.daal.data_management.data.Factory;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTI_CLASS_CLASSIFIER__MODEL"></a>
 * @brief Model of the classifier trained by the multi_class_classifier.training.TrainingBatch algorithm.
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
/** @} */
