/* file: TrainingBatch.java */
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
 * @defgroup stump_training_batch Batch
 * @ingroup stump_training
 * @{
 */
package com.intel.daal.algorithms.stump.training;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.classifier.training.TrainingInput;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__STUMP__TRAINING__TRAININGBATCH"></a>
 * @brief Trains the decision stump model
 */
public class TrainingBatch extends com.intel.daal.algorithms.weak_learner.training.TrainingBatch {
    public TrainingMethod method;   /*!< %Training method for the algorithm */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the decision stump training algorithm by copying input objects
     * of another decision stump training algorithm
     * @param context   Context to manage the stump training algorithm
     * @param other     An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    public TrainingBatch(DaalContext context, TrainingBatch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), this.method.getValue());
        input = new TrainingInput(getContext(), cObject, ComputeMode.batch);
    }

    /**
     * Constructs the decision stump training algorithm
     * @param context   Context to manage created stump training algorithm
     * @param cls       Data type to use in intermediate computations for the decision stump training algorithm,
     *                  Double.class or Float.class
     * @param method    the decision stump training method, @ref TrainingMethod
     */
    public TrainingBatch(DaalContext context, Class<? extends Number> cls, TrainingMethod method) {
        super(context);

        this.method = method;

        if (this.method != TrainingMethod.defaultDense) {
            throw new IllegalArgumentException("method unsupported");
        }

        if (cls != Double.class && cls != Float.class) {
            throw new IllegalArgumentException("type unsupported");
        }

        if (cls == Double.class) {
            prec = Precision.doublePrecision;
        } else {
            prec = Precision.singlePrecision;
        }

        this.cObject = cInit(prec.getValue(), this.method.getValue());
        input = new TrainingInput(getContext(), cObject, ComputeMode.batch);
    }

    /**
     * Returns the newly allocated decision stump training algorithm
     * with a copy of input objects and parameters of this decision stump training algorithm
     * @param context   Context to manage the stump training algorithm
     *
     * @return The newly allocated algorithm
     */
    @Override
    public TrainingBatch clone(DaalContext context) {
        return new TrainingBatch(context, this);
    }

    private native long cInit(int prec, int method);

    private native long cClone(long algAddr, int prec, int method);
}
/** @} */
