/* file: TrainingResult.java */
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
 * @ingroup multi_class_classifier_training
 * @{
 */
package com.intel.daal.algorithms.multi_class_classifier.training;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.classifier.training.TrainingResultId;
import com.intel.daal.algorithms.multi_class_classifier.Model;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTI_CLASS_CLASSIFIER__TRAINING__TRAININGRESULT"></a>
 * @brief Provides methods to access the results obtained with the compute() method of multi_class_classifier.training.TrainingBatch algorithm
 */
public final class TrainingResult extends com.intel.daal.algorithms.classifier.training.TrainingResult {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public TrainingResult(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Returns final result of the multi-class classifier algorithm
     * @param id   Identifier of the result, @ref classifier.training.TrainingResultId
     * @return         Result that corresponds to the given identifier
     */
    public Model get(TrainingResultId id) {
        if (id == TrainingResultId.model) {
            return new Model(getContext(), cGetModel(cObject, TrainingResultId.model.getValue()));
        } else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    private native long cGetModel(long resAddr, int id);
}
/** @} */
