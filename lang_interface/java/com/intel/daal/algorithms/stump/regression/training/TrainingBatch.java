/* file: TrainingBatch.java */
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
 * @defgroup stump_training_batch Batch
 * @ingroup stump_training
 * @{
 */
package com.intel.daal.algorithms.stump.regression.training;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.stump.regression.Parameter;
import com.intel.daal.algorithms.regression.training.TrainingInput;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__STUMP__REGRESSION__TRAINING__TRAININGBATCH"></a>
 * @brief Trains the decision stump model
 */
public class TrainingBatch extends com.intel.daal.algorithms.regression.training.TrainingBatch {
    public TrainingMethod method;   /*!< %Training method for the algorithm */
    public Parameter parameter;

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
        this.prec      = other.prec;
        this.method    = other.method;
        this.cObject   = cClone(other.cObject, prec.getValue(), method.getValue());
        this.input     = new TrainingInput(getContext(), this.cObject, ComputeMode.batch);
        this.parameter = new Parameter( getContext(), cInitParameter(this.cObject,
                                                                     this.prec.getValue(),
                                                                     this.method.getValue()) );
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
        init(cls, method);
    }

    /**
     * Constructs the decision stump training algorithm
     * @param context   Context to manage created stump training algorithm
     */
    public TrainingBatch(DaalContext context) {
        super(context);
        init(Float.class, TrainingMethod.defaultDense);
    }

    @Override
    public TrainingResult compute() {
        super.compute();
        return new TrainingResult(getContext(), cGetResult(this.cObject,
                                                           this.prec.getValue(),
                                                           this.method.getValue()));
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

    private void init(Class<? extends Number> cls, TrainingMethod method) {
        if (cls != Double.class && cls != Float.class) {
            throw new IllegalArgumentException("type unsupported");
        }

        final Precision prec = (cls == Double.class)
            ? Precision.doublePrecision
            : Precision.singlePrecision;
        init(prec, method);
    }

    private void init(Precision prec, TrainingMethod method) {
        if (method != TrainingMethod.defaultDense) {
            throw new IllegalArgumentException("method unsupported");
        }

        this.prec      = prec;
        this.method    = method;
        this.cObject   = cInit(prec.getValue(), method.getValue());
        this.input     = new TrainingInput(getContext(), this.cObject, ComputeMode.batch);
        this.parameter = new Parameter( getContext(), cInitParameter(this.cObject,
                                                                     prec.getValue(),
                                                                     method.getValue()) );
    }

    private native long cInit(int prec, int method);
    private native long cInitParameter(long selfPtr, int prec, int method);
    private native long cGetResult(long selfPtr, int prec, int method);
    private native long cClone(long selfPtr, int prec, int method);
}
/** @} */
