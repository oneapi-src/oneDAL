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
 * @ingroup neural_networks_training
 * @{
 */
package com.intel.daal.algorithms.neural_networks.training;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__TRAINING__TRAININGRESULT"></a>
 * @brief Provides methods to access result obtained with the compute() method of the neural network training algorithm
 */
public class TrainingResult extends com.intel.daal.algorithms.Result {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the result of the neural network training algorithm
     * @param context   Context to manage the result of the neural network training algorithm
     */
    public TrainingResult(DaalContext context) {
        super(context);
        cObject = cNewResult();
    }

    public TrainingResult(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Returns the result of the neural network model based training
     * @param  id   Identifier of the result
     * @return Result that corresponds to the given identifier
     */
    public TrainingModel get(TrainingResultId id) {
        if (id == TrainingResultId.model) {
            return new TrainingModel(getContext(), cGetValue(cObject, id.getValue()));
        }
        else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
     * Sets the result of neural network model based training
     * @param id   Identifier of the result
     * @param val  Result that corresponds to the given identifier
     */
    public void set(TrainingResultId id, TrainingModel val) {
        if (id == TrainingResultId.model) {
            cSetValue(cObject, id.getValue(), val.getCObject());
        }
        else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    private native long cNewResult();

    private native long cGetValue(long cObject, int id);

    private native void cSetValue(long cObject, int id, long modelAddr);
}
/** @} */
