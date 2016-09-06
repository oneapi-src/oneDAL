/* file: Batch.java */
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
 * @brief Contains classes to run the quantile algorithms
 */
package com.intel.daal.algorithms.quantiles;

import com.intel.daal.algorithms.AnalysisBatch;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__QUANTILES__BATCH"></a>
 * @brief Computes values of quantiles in the batch processing mode.
 * \n<a href="DAAL-REF-QUANTILES-ALGORITHM">Quantiles algorithm description and usage models</a>
 *
 * @tparam algorithmFPType  Data type to use in intermediate computations for the quantile algorithms, double or float
 * @tparam method           Quantiles computation method, @ref daal::algorithms::quantiles::Method
 *
 * @par Enumerations
 *      - @ref Method   Quantiles computation methods
 *      - @ref InputId  Identifiers of quantiles input objects
 *      - @ref ResultId Identifiers of quantiles results
 *
 * @par References
 *      - Input class
 *      - Parameter class
 *      - Result class
 */
public class Batch extends AnalysisBatch {
    public Input          input;     /*!< %Input data */
    public Parameter  parameter;     /*!< Parameters of the algorithm */
    public Method     method; /*!< Computation method for the algorithm */
    private Precision                 prec; /*!< Precision of intermediate computations */

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs algorithm that computes quantiles by copying input objects and parameters
     * of another algorithm
     * @param context      Context to manage the quantile algorithms
     * @param other        An algorithm to be used as the source to initialize the input objects
     *                     and parameters of the algorithm
     */
    public Batch(DaalContext context, Batch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), this.method.getValue());

        input = new Input(getContext(), cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new Parameter(getContext(), cInitParameter(this.cObject, prec.getValue(), method.getValue()));
    }

    /**
     * <a name="DAAL-METHOD-ALGORITHM__QUANTILES__BATCH__BATCH"></a>
     *  @brief Computes quantiles in the batch processing mode
     * @param context      Context to manage the quantile algorithms
     * @param cls          Data type to use in intermediate computations for the quantile algorithms, Double.class or Float.class
     * @param method       Quantiles computation methods, @ref Method
     */
    public Batch(DaalContext context, Class<? extends Number> cls, Method method) {
        super(context);
        this.method = method;
        if (cls != Double.class && cls != Float.class) {
            throw new IllegalArgumentException("type unsupported");
        }

        if (this.method != Method.defaultDense) {
            throw new IllegalArgumentException("method unsupported");
        }

        if (cls == Double.class) {
            prec = Precision.doublePrecision;
        } else {
            prec = Precision.singlePrecision;
        }

        this.cObject = cInit(prec.getValue(), this.method.getValue());

        input = new Input(getContext(), cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new Parameter(getContext(), cInitParameter(this.cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Runs the quantile algorithm
     * @return  Quantiles computation results
     */
    @Override
    public Result compute() {
        super.compute();
        Result result = new Result(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
        return result;
    }

    /**
     * Registers user-allocated memory to store results of the quantile algorithms
     * @param result    Structure to store results of the quantile algorithms
     */
    public void setResult(Result result) {
        cSetResult(cObject, prec.getValue(), method.getValue(), result.getCObject());
    }

    /**
     * Returns the newly allocated algorithm that computes quantiles
     * with a copy of input objects and parameters of this algorithm
     * @param context      Context to manage the quantile algorithms
     *
     * @return The newly allocated algorithm
     */
    @Override
    public Batch clone(DaalContext context) {
        return new Batch(context, this);
    }

    private native long cInit(int prec, int method);

    private native long cInitParameter(long algAddr, int prec, int method);

    private native long cGetInput(long algAddr, int prec, int method);

    private native long cGetResult(long algAddr, int prec, int method);

    private native void cSetResult(long cObject, int prec, int method, long cResult);

    private native long cClone(long algAddr, int prec, int method);
}
