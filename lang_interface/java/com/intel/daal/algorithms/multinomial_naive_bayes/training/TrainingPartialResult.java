/* file: TrainingPartialResult.java */
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
 * @ingroup multinomial_naive_bayes_training
 * @{
 */
package com.intel.daal.algorithms.multinomial_naive_bayes.training;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.classifier.training.PartialResultId;
import com.intel.daal.algorithms.multinomial_naive_bayes.PartialModel;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTINOMIAL_NAIVE_BAYES__TRAINING__TRAININGPARTIALRESULT"></a>
 * @brief Provides methods to access results obtained with the compute() method of the
 *        naive Bayes training algorithm in the online or distributed processing mode
 */

public final class TrainingPartialResult extends com.intel.daal.algorithms.classifier.training.TrainingPartialResult {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the partial result of the naive Bayes training algorithm in the online or distributed processing mode
     * @param context   Context to manage the partial result of the naive Bayes training algorithm in the online or distributed processing mode
     */
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
/** @} */
