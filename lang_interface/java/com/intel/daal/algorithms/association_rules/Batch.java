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

package com.intel.daal.algorithms.association_rules;

import com.intel.daal.algorithms.AnalysisBatch;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__ASSOCIATION_RULES__BATCH"></a>
 * @brief Computes the result of the association rules algorithm in the batch processing mode.
 * \n<a href="DAAL-REF-ASSOCIATION_RULES-ALGORITHM">Association rules algorithm description and usage models</a>
 *
 * @par References
 *      - @ref Method class
 *      - @ref Parameter class
 *      - @ref InputId class
 *      - @ref ResultId class
 *      - @ref Input class
 *      - @ref Result class
 *      - @ref RulesOrderId class
 *      - @ref ItemsetsOrderId class
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
     * Constructs the association rules algorithm by copying input objects and parameters
     * of another association rules algorithm
     *
     * @param context      Context to manage the association rules algorithm
     * @param other        An algorithm to be used as the source to initialize the input objects
     *                     and parameters of the algorithm
     */
    public Batch(DaalContext context, Batch other) {
        super(context);
        method = other.method;
        prec = other.prec;
        this.cObject = cClone(other.cObject, prec.getValue(), method.getValue());
        input = new Input(getContext(), cObject, prec, method, ComputeMode.batch);
        parameter = new Parameter(getContext(),
                cInitParameter(this.cObject, prec.getValue(), method.getValue(), ComputeMode.batch.getValue()));
    }

    /**
     * <a name="DAAL-METHOD-ALGORITHMS__ASSOCIATION_RULES__BATCH__BATCH"></a>
     * Constructs the association rules algorithm
     *
     * @param context      Context to manage the association rules algorithm
     * @param cls          Data type to use in intermediate computations for the association rules algorithm, Double.class or Float.class
     * @param method       Association rules computation method, @ref Method
     */
    public Batch(DaalContext context, Class<? extends Number> cls, Method method) {
        super(context);
        this.method = method;
        if (cls != Double.class && cls != Float.class) {
            throw new IllegalArgumentException("type unsupported");
        }

        if (this.method != Method.apriori) {
            throw new IllegalArgumentException("method unsupported");
        }

        if (cls == Double.class) {
            prec = Precision.doublePrecision;
        } else {
            prec = Precision.singlePrecision;
        }

        this.cObject = cInit(prec.getValue(), this.method.getValue());

        input = new Input(getContext(), cObject, prec, method, ComputeMode.batch);
        parameter = new Parameter(getContext(),
                cInitParameter(this.cObject, prec.getValue(), method.getValue(), ComputeMode.batch.getValue()));
    }

    /**
     * Computes results of the association rules algorithm
     * @return Results of the association rules algorithm
     */
    @Override
    public Result compute() {
        super.compute();
        Result result = new Result(getContext(), cObject, prec, method, ComputeMode.batch);
        return result;
    }

    /**
     * Registers user-allocated memory to store results of the association rules algorithm
     * @param result  Structure to store results of the association rules algorithm
     */
    public void setResult(Result result) {
        cSetResult(cObject, prec.getValue(), method.getValue(), result.getCObject());
    }

    /**
     * Returns the newly allocated association rules algorithm with a copy of input objects
     * and parameters of this association rules algorithm
     * @param context      Context to manage the association rules algorithm
     *
     * @return The newly allocated algorithm
     */
    @Override
    public Batch clone(DaalContext context) {
        return new Batch(context, this);
    }

    private native long cInit(int prec, int method);

    private native long cInitParameter(long algAddr, int prec, int method, int cmode);

    private native void cSetResult(long cObject, int prec, int method, long cResult);

    private native long cClone(long algAddr, int prec, int method);
}
