/* file: TrainingDistributedInput.java */
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
 * @ingroup multinomial_naive_bayes_training_distributed
 * @{
 */
package com.intel.daal.algorithms.multinomial_naive_bayes.training;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__CLASSIFIER__MULTINOMIAL_NAIVE_BAYES__TRAININGDISTRIBUTEDINPUT"></a>
 * @brief  Input objects of the naive Bayes model training algorithm
 *         in the distributed computing mode
 */
public class TrainingDistributedInput extends com.intel.daal.algorithms.Input {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public TrainingDistributedInput(DaalContext context, long cInput) {
        super(context, cInput);
    }

    /**
     * Adds input objects to the classifier model training algorithm on the master node
     * @param id      Identifier of the input object
     * @param val     Value of the input object
     */
    public void add(TrainingDistributedInputId id, TrainingPartialResult val) {
        if (id != TrainingDistributedInputId.partialModels) {
            throw new IllegalArgumentException("Incorrect TrainingDistributedInputId");
        }
        cAddInput(this.cObject, id.getValue(), val.getCObject());
    }

    private native void cAddInput(long cObject, int id, long presAddr);
}
/** @} */
