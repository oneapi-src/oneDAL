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
 * @ingroup logistic_regression_training
 * @{
 */
package com.intel.daal.algorithms.logistic_regression.training;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.classifier.training.TrainingResultId;
import com.intel.daal.algorithms.logistic_regression.Model;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LOGISTIC_REGRESSION__TRAINING__TRAININGRESULT"></a>
 * @brief Provides methods to access the result of logistic regression model-based training
 */
public final class TrainingResult extends com.intel.daal.algorithms.classifier.training.TrainingResult {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the logistic regression model-based training result
     * @param context   Context to manage logistic regression training result
     */
    public TrainingResult(DaalContext context) {
        super(context);
        cObject = cNewResult();
    }

    public TrainingResult(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Returns the result of logistic regression model-based training
     * @param id    Identifier of the result
     * @return      Result that corresponds to the given identifier
     */
    public Model get(TrainingResultId id) {
        if (id != TrainingResultId.model) {
            throw new IllegalArgumentException("id unsupported");
        }
        return new Model(getContext(), cGetModel(cObject, TrainingResultId.model.getValue()));
    }

    private native long cNewResult();
    private native long cGetModel(long resAddr, int id);

}
/** @} */
