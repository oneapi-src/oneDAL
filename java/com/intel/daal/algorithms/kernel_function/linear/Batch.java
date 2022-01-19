/* file: Batch.java */
/*******************************************************************************
* Copyright 2014-2022 Intel Corporation
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
 * @brief Contains classes for computing the linear kernel function
 */
/**
 * @defgroup kernel_function_linear Linear Kernel
 * @ingroup kernel_function
 * @{
 */
/**
 * @defgroup kernel_function_linear_batch Batch
 * @ingroup kernel_function_linear
 * @{
 */
package com.intel.daal.algorithms.kernel_function.linear;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KERNEL_FUNCTION__LINEAR__BATCH"></a>
 * @brief Computes the linear kernel function in the batch processing mode
 * <!-- \n<a href="DAAL-REF-KERNEL_FUNCTION-ALGORITHM">Kernel function algorithm description and usage models</a> -->
 *
 * \par References
 *      - Parameter class
 */
public class Batch extends com.intel.daal.algorithms.kernel_function.Batch {
    public Input      input;     /*!< %Input data */
    public Method     method;    /*!< Computation method for the algorithm */
    public Parameter  parameter; /*!< Parameters of the algorithm */
    private Precision prec;      /*!< Precision of intermediate computations */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the linear kernel function algorithm by copying input objects and parameters
     * of another linear kernel function algorithm
     * @param context  Context to manage the linear kernel function
     * @param other    An algorithm to be used as the source to initialize the input objects
     *                 and parameters of the algorithm
     */
    public Batch(DaalContext context, Batch other) {
        super(context);

        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), method.getValue());
        input = new Input(getContext(), cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new Parameter(context, cGetParameter(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs the linear kernel function algorithm
     * @param context  Context to manage the linear kernel function
     * @param cls      Data type to use in intermediate computations of the linear kernel function, Double.class or Float.class
     * @param method   Linear kernel function computation method, @ref Method
     */
    public Batch(DaalContext context, Class<? extends Number> cls, Method method) {
        super(context);

        this.method = method;

        if (method != Method.defaultDense && method != Method.fastCSR) {
            throw new IllegalArgumentException("method unsupported");
        }
        if (cls != Double.class && cls != Float.class) {
            throw new IllegalArgumentException("type unsupported");
        }

        if (cls == Double.class) {
            prec = Precision.doublePrecision;
        }
        else {
            prec = Precision.singlePrecision;
        }

        this.cObject = cInit(prec.getValue(), method.getValue());
        input = new Input(getContext(), cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new Parameter(context, cGetParameter(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs the linear kernel function algorithm with default method
     * @param context  Context to manage the linear kernel function
     * @param cls Data type to use in intermediate computations of the linear kernel function, Double.class or Float.class
     */
    public Batch(DaalContext context, Class<? extends Number> cls) {
        super(context);

        this.method = Method.defaultDense;

        if (cls != Double.class && cls != Float.class) {
            throw new IllegalArgumentException("type unsupported");
        }

        if (cls == Double.class) {
            prec = Precision.doublePrecision;
        }
        else {
            prec = Precision.singlePrecision;
        }

        this.cObject = cInit(prec.getValue(), method.getValue());
        input = new Input(getContext(), cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new Parameter(context, cGetParameter(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Computes the linear kernel function
     * @return Structure that contains the computed linear kernel function
     */
    @Override
    public Result compute() {
        super.compute();
        return new Result(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Registers user-allocated memory to store results of the linear kernel function
     * @param result  Structure to store results of the linear kernel function
     */
    public void setResult(Result result) {
        cSetResult(cObject, prec.getValue(), method.getValue(), result.getCObject());
    }

    /**
     * Returns the newly allocated linear kernel function algorithm with a copy of input objects
     * and parameters of this linear kernel function algorithm
     * @param context  Context to manage the linear kernel function
     *
     * @return The newly allocated algorithm
     */
    @Override
    public Batch clone (DaalContext context) {
        return new Batch(context, this);
    }

    private native long cInit(int prec, int method);

    private native long cGetParameter(long cAlgorithm, int prec, int method);

    private native void cSetResult(long algAddr, int prec, int method, long resAddr);

    private native long cGetResult(long algAddr, int prec, int method);

    private native long cGetInput(long algAddr, int prec, int method);

    private native long cClone(long algAddr, int prec, int method);
}
/** @} */
/** @} */
