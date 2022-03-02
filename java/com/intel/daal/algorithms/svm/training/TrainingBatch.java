/* file: TrainingBatch.java */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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
 * @defgroup svm_training_batch Batch
 * @ingroup svm_training
 * @{
 */
/**
 * @brief Contains classes to train the SVM model
 */
package com.intel.daal.algorithms.svm.training;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.classifier.training.TrainingInput;
import com.intel.daal.algorithms.svm.Parameter;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__SVM__TRAINING__TRAININGBATCH"></a>
 * @brief %Algorithm class to train the SVM model
 * <!-- \n<a href="DAAL-REF-SVM-ALGORITHM">SVM algorithm description and usage models</a> -->
 *
 * @par References
 *      - com.intel.daal.algorithms.svm.Model class
 *      - com.intel.daal.algorithms.classifier.training.InputId class
 *      - com.intel.daal.algorithms.classifier.training.TrainingResultId class
 *      - com.intel.daal.algorithms.classifier.training.TrainingInput class
 */
public class TrainingBatch extends com.intel.daal.algorithms.classifier.training.TrainingBatch {
    public Parameter  parameter;     /*!< Parameters of the algorithm */
    public TrainingMethod method;   /*!< %Training method for the algorithm */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the SVM training algorithm by copying input objects and parameters
     * of another SVM training algorithm
     * @param context   Context to manage created SVM training algorithm
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
     * Constructs the SVM training algorithm
     * @param context   Context to manage created SVM training algorithm
     * @param cls       Data type to use in intermediate computations for the SVM training algorithm,
     *                  Double.class or Float.class
     * @param method    SVM training method, @ref TrainingMethod
     */
    public TrainingBatch(DaalContext context, Class<? extends Number> cls, TrainingMethod method) {
        super(context);
        this.method = method;
        if (cls != Double.class && cls != Float.class) {
            throw new IllegalArgumentException("type unsupported");
        }

        if (this.method != TrainingMethod.boser && this.method != TrainingMethod.thunder) {
            throw new IllegalArgumentException("method unsupported");
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
     * Trains the model of the SVM algorithm
     * @return Structure that contains results of the SVM training algorithm
     */
    @Override
    public TrainingResult compute() {
        super.compute();
        TrainingResult result = new TrainingResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
        return result;
    }

    /**
     * Returns the newly allocated SVM training algorithm with a copy of input objects
     * and parameters of this SVM training algorithm
     * @param context   Context to manage created SVM training algorithm
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
