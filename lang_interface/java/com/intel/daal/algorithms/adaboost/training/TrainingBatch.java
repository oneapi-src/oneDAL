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
 * @defgroup adaboost_training_batch Batch
 * @ingroup adaboost_training
 * @{
 */
/**
 * @brief Contains classes for training AdaBoost models
 */
package com.intel.daal.algorithms.adaboost.training;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.adaboost.Parameter;
import com.intel.daal.algorithms.classifier.training.TrainingInput;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__ADABOOST__TRAINING__TRAININGBATCH"></a>
 * @brief Trains a model of the AdaBoost algorithm in the batch processing mode
 * <!-- \n<a href="DAAL-REF-ADABOOST-ALGORITHM">AdaBoost algorithm description and usage models</a> -->
 *
 * \par References
 *      - com.intel.daal.algorithms.classifier.training.InputId class
 *      - com.intel.daal.algorithms.classifier.training.TrainingResultId class
 *      - com.intel.daal.algorithms.adaboost.Model class
 *      - com.intel.daal.algorithms.classifier.training.TrainingInput class
 */
public class TrainingBatch extends com.intel.daal.algorithms.classifier.training.TrainingBatch {
    public TrainingMethod method;    /*!< %Training method for the algorithm */
    public Parameter      parameter; /*!< Parameters of the algorithm */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the AdaBoost training algorithm by copying input objects and parameters
     * of another AdaBoost training algorithm
     * @param context   Context to manage AdaBoost training
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
     * Constructs the AdaBoost training algorithm
     * @param context   Context to manage AdaBoost training
     * @param nClasses  Number of classes
     * @param cls       Data type to use in intermediate computations for AdaBoost training,
     *                  Double.class or Float.class
     * @param method    AdaBoost training method, @ref TrainingMethod
     */
    public TrainingBatch(DaalContext context, long nClasses, Class<? extends Number> cls, TrainingMethod method) {
        super(context);
        init(Precision.fromClass(cls), method, nClasses);
    }

    /**
     * Constructs the AdaBoost training algorithm
     * @param context   Context to manage AdaBoost training
     * @param nClasses  Number of classes
     */
    public TrainingBatch(DaalContext context, long nClasses) {
        super(context);
        init(Precision.singlePrecision, TrainingMethod.defaultDense, nClasses);
    }

    /**
     * Trains a model of the AdaBoost algorithm
     * @return Structure that contains results of the AdaBoost training algorithm
     */
    @Override
    public TrainingResult compute() {
        super.compute();
        TrainingResult result = new TrainingResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
        return result;
    }

    /**
     * Returns the newly allocated AdaBoost training algorithm with a copy of input objects
     * and parameters of this AdaBoost training algorithm
     * @param context   Context to manage AdaBoost training
     *
     * @return The newly allocated algorithm
     */
    @Override
    public TrainingBatch clone(DaalContext context) {
        return new TrainingBatch(context, this);
    }

    private void init(Precision prec, TrainingMethod method, Long nClasses) {
        this.prec      = prec;
        this.method    = method;
        this.cObject   = cInit(prec.getValue(), method.getValue(), nClasses);
        this.input     = new TrainingInput(getContext(), this.cObject, ComputeMode.batch);
        this.parameter = new Parameter( getContext(), cInitParameter(this.cObject,
                                                                     prec.getValue(),
                                                                     method.getValue()) );
    }

    private native long cInit(int prec, int method, long nClasses);

    private native long cInitParameter(long algAddr, int prec, int method);

    private native long cGetResult(long algAddr, int prec, int method);

    private native long cClone(long algAddr, int prec, int method);
}
/** @} */
