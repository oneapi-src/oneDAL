/* file: TrainingResult.java */
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
 * @defgroup boosting_training Training
 * @brief Contains classes for training boosting models
 * @ingroup boosting
 * @{
 */
package com.intel.daal.algorithms.boosting.training;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.boosting.Model;
import com.intel.daal.algorithms.classifier.training.TrainingResultId;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__BOOSTING__TRAINING__TRAININGRESULT"></a>
 * @brief Provides methods to access final results obtained with the compute() method of a %boosting training algorithm in the batch processing mode
 */

public class TrainingResult extends com.intel.daal.algorithms.classifier.training.TrainingResult {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the result of boosting training algorithm
     * @param context   Context to manage the result of boosting training algorithm
     */
    public TrainingResult(DaalContext context) {
        super(context);
    }

    public TrainingResult(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Returns the final result of the %boosting training algorithm
     * @param id   Identifier of the result
     * @return         %Result that corresponds to the given identifier
     */
    public Model get(TrainingResultId id) {
        if (id == TrainingResultId.model) {
            return new Model(getContext(), cGetModel(cObject, TrainingResultId.model.getValue()));
        } else {
            return null;
        }
    }

    protected native long cGetModel(long resAddr, int id);
}
/** @} */
