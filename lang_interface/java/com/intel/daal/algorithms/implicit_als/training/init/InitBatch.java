/* file: InitBatch.java */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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
 * @brief Contains classes for the implicit ALS initialization algorithm
 */
package com.intel.daal.algorithms.implicit_als.training.init;

import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.TrainingBatch;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__INIT__INITBATCH"></a>
 * @brief Computes the initial model for the implicit ALS algorithm in the batch processing mode
 *
 * \par References
 *      - @ref InitParameter class
 *      - @ref InitInput class
 *      - @ref InitMethod class
 *      - @ref InitResult class
 *
 */
public class InitBatch extends TrainingBatch {
    public InitParameter parameter;   /*!< Parameters for the initialization algorithm */
    public InitInput   input;           /*!< %Input data */
    public InitMethod    method;  /*!< Initialization method for the algorithm */
    private InitResult result; /*!< %Result of the initialization algorithm */
    private Precision precision; /*!< Precision of intermediate computations */

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs an algorithm for computing initial values for the implicit ALS algorithm in the batch processing mode
     * by copying input objects and parameters of another algorithm for computing initial values for the implicit ALS algorithm
     *
     * @param context   Context to manage the implicit ALS algorithm
     * @param other     An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    public InitBatch(DaalContext context, InitBatch other) {
        super(context);
        this.method = other.method;
        precision = other.precision;

        this.cObject = cClone(other.cObject, precision.getValue(), method.getValue());

        input = new InitInput(getContext(), cObject, precision, method, ComputeMode.batch);
        parameter = new InitParameter(getContext(),
                cInitParameter(this.cObject, precision.getValue(), method.getValue()));
    }

    /**
     * <a name="DAAL-METHOD-ALGORITHM__IMPLICIT_ALS__TRAINING__INIT__INITBATCH__INITBATCH"></a>
     * Constructs an algorithm for computing initial values for the implicit ALS algorithm in the batch processing mode
     *
     * @param context   Context to manage the implicit ALS algorithm
     * @param cls       Data type to use in intermediate computations for the implicit ALS algorithm,
     *                  Double.class or Float.class
     * @param method    Implicit ALS initialization method, @ref TrainingMethod
     */
    public InitBatch(DaalContext context, Class<? extends Number> cls, InitMethod method) {
        super(context);
        this.method = method;

        if (this.method != InitMethod.fastCSR && this.method != InitMethod.defaultDense) {
            throw new IllegalArgumentException("method unsupported");
        }

        if (cls == Double.class) {
            precision = Precision.doublePrecision;
        } else if (cls == Float.class) {
            precision = Precision.singlePrecision;
        } else {
            throw new IllegalArgumentException("type unsupported");
        }

        this.cObject = cInit(precision.getValue(), method.getValue());

        input = new InitInput(getContext(), cObject, precision, method, ComputeMode.batch);
        parameter = new InitParameter(getContext(),
                cInitParameter(this.cObject, precision.getValue(), method.getValue()));
    }

    /**
    * Computes the initial model for the implicit ALS algorithm in the batch processing mode
    * @return Computed initial model
    */
    @Override
    public InitResult compute() {
        super.compute();
        InitResult result = new InitResult(getContext(), cObject, precision, method, ComputeMode.batch);
        return result;
    }

    /**
    * Registers user-allocated memory for storing computed initial values for the implicit ALS algorithm
    * @param result Structure for storing computed initial values for the implicit ALS algorithm
    */
    public void setResult(InitResult result) {
        cSetResult(cObject, precision.getValue(), method.getValue(), result.getCObject());
    }

    /**
     * Returns the newly allocated algorithm for computing initial values for the implicit ALS algorithm
     * in the batch processing mode with a copy of input objects of this algorithm for computing initial values
     * for the implicit ALS algorithm
     * @param context   Context to manage the implicit ALS algorithm
     *
     * @return The newly allocated algorithm
     */
    @Override
    public InitBatch clone(DaalContext context) {
        return new InitBatch(context, this);
    }

    private native long cInit(int precision, int method);

    private native long cInitParameter(long algAddr, int prec, int method);

    private native void cSetResult(long cObject, int prec, int method, long cResult);

    private native long cClone(long algAddr, int prec, int method);
}
