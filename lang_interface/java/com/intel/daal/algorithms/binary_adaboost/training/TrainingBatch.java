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
 * @defgroup binary_adaboost_training_batch Batch
 * @ingroup binary_adaboost_training
 * @{
 */
/**
 * @brief Contains classes for training binary_adaboost models
 */
package com.intel.daal.algorithms.binary_adaboost.training;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.binary_adaboost.Parameter;
import com.intel.daal.algorithms.classifier.training.TrainingInput;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__binary_adaboost__TRAINING__TRAININGBATCH"></a>
 * @brief Trains a model of the binary_adaboost algorithm in the batch processing mode
 * <!-- \n<a href="DAAL-REF-binary_adaboost-ALGORITHM">binary_adaboost algorithm description and usage models</a> -->
 *
 * \par References
 *      - com.intel.daal.algorithms.classifier.training.InputId class
 *      - com.intel.daal.algorithms.classifier.training.TrainingResultId class
 *      - com.intel.daal.algorithms.binary_adaboost.Model class
 *      - com.intel.daal.algorithms.classifier.training.TrainingInput class
 */
public class TrainingBatch extends com.intel.daal.algorithms.boosting.training.TrainingBatch {
    public TrainingMethod method;    /*!< %Training method for the algorithm */
    public Parameter      parameter; /*!< Parameters of the algorithm */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the binary_adaboost training algorithm by copying input objects and parameters
     * of another binary_adaboost training algorithm
     * @param context   Context to manage binary_adaboost training
     * @param other     An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    public TrainingBatch(DaalContext context, TrainingBatch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;
        this.cObject = cClone(other.cObject, prec.getValue(), this.method.getValue());
        input = new TrainingInput(getContext(), cObject, ComputeMode.batch);
        parameter = new Parameter(getContext(), cInitParameter(this.cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs the binary_adaboost training algorithm
     * @param context   Context to manage binary_adaboost training
     * @param cls       Data type to use in intermediate computations for binary_adaboost training,
     *                  Double.class or Float.class
     * @param method    binary_adaboost training method, @ref TrainingMethod
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
        parameter = new Parameter(getContext(), cInitParameter(this.cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Trains a model of the binary_adaboost algorithm
     * @return Structure that contains results of the binary_adaboost training algorithm
     */
    @Override
    public TrainingResult compute() {
        super.compute();
        TrainingResult result = new TrainingResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
        return result;
    }

    /**
     * Returns the newly allocated binary_adaboost training algorithm with a copy of input objects
     * and parameters of this binary_adaboost training algorithm
     * @param context   Context to manage binary_adaboost training
     *
     * @return The newly allocated algorithm
     */
    @Override
    public TrainingBatch clone(DaalContext context) {
        return new TrainingBatch(context, this);
    }

    private native long cInit(int prec, int method);

    private native long cInitParameter(long algAddr, int prec, int method);

    private native long cGetResult(long algAddr, int prec, int method);

    private native long cClone(long algAddr, int prec, int method);
}
/** @} */
