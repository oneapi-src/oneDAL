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
 * @defgroup gbt_regression_training_batch Batch
 * @ingroup gbt_regression_training
 * @{
 */
/**
 * @brief Contains classes for training gradient boosted trees regression models
 */
package com.intel.daal.algorithms.gbt.regression.training;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.gbt.regression.training.TrainingInput;
import com.intel.daal.algorithms.gbt.regression.training.Parameter;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__REGRESSION__TRAINING__TRAININGBATCH"></a>
 * @brief Trains a model of the gradient boosted trees regression algorithm in the batch processing mode
 * <!-- \n<a href="DAAL-REF-GBT__REGRESSION-ALGORITHM">gradient boosted trees regression algorithm description and usage models</a> -->
 *
 * \par References
 *      - com.intel.daal.algorithms.gbt.regression.training.InputId class
 *      - com.intel.daal.algorithms.gbt.regression.training.TrainingResultId class
 *      - com.intel.daal.algorithms.gbt.regression.Model class
 *      - com.intel.daal.algorithms.gbt.regression.training.TrainingInput class
 */
public class TrainingBatch extends com.intel.daal.algorithms.TrainingBatch {
    protected Precision  prec;
    public TrainingMethod method; /*!< %Training method for the algorithm */
    public Parameter  parameter; /*!< Parameters of the algorithm */
    public TrainingInput input; /*!< Input of the algorithm */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the gradient boosted trees regression training algorithm by copying input objects and parameters
     * of another gradient boosted trees regression training algorithm
     * @param context   Context to manage gradient boosted trees regression training
     * @param other     An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    public TrainingBatch(DaalContext context, TrainingBatch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), this.method.getValue());
        input = new TrainingInput(getContext(), cObject, prec, method);
        parameter = new Parameter(getContext(),
                cInitParameter(this.cObject, prec.getValue(), method.getValue(), ComputeMode.batch.getValue()));
    }

    /**
     * Constructs the gradient boosted trees regression training algorithm
     * @param context   Context to manage gradient boosted trees regression training
     * @param cls       Data type to use in intermediate computations for gradient boosted trees regression training,
     *                  Double.class or Float.class
     * @param method    gradient boosted trees regression training method, @ref TrainingMethod
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
        input = new TrainingInput(getContext(), cObject, prec, method);
        parameter = new Parameter(getContext(),
                cInitParameter(this.cObject, prec.getValue(), method.getValue(), ComputeMode.batch.getValue()));
    }

    /**
     * Trains a model of the gradient boosted trees regression algorithm
     * @return Structure that contains results of the gradient boosted trees regression training algorithm
     */
    @Override
    public TrainingResult compute() {
        super.compute();
        TrainingResult result = new TrainingResult(getContext(), cObject, prec, ComputeMode.batch);
        return result;
    }

    /**
     * Returns the newly allocated gradient boosted trees regression training algorithm with a copy of input objects
     * and parameters of this gradient boosted trees regression training algorithm
     * @param context   Context to manage gradient boosted trees regression training
     *
     * @return The newly allocated algorithm
     */
    @Override
    public TrainingBatch clone(DaalContext context) {
        return new TrainingBatch(context, this);
    }

    private native long cInit(int prec, int method);

    private native long cInitParameter(long algAddr, int prec, int method, int cmode);

    private native long cClone(long algAddr, int prec, int method);
}
/** @} */
