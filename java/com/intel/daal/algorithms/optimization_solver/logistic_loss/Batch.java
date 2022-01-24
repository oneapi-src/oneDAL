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
 * @defgroup logistic_loss Logistic loss objective function
 * @brief Contains classes for computing the Logistic loss objective function
 * @ingroup objective_function
 * @{
 */
/**
 * @defgroup logistic_loss_batch Batch
 * @ingroup logistic_loss
 * @{
 */
package com.intel.daal.algorithms.optimization_solver.logistic_loss;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;
import com.intel.daal.algorithms.optimization_solver.objective_function.Result;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__LOGISTIC_LOSS__BATCH"></a>
 * @brief Computes the logistic loss objective function in the batch processing mode
 * <!-- \n<a href="DAAL-REF-LOGISTIC_LOSS-ALGORITHM">The logistic loss objective function algorithm description and usage models</a> -->
 *
 * @par References
 *      - Parameter class
 *      - InputId class
 *      - com.intel.daal.algorithms.optimization_solver.objective_function.ResultId class
 *      - \ref com.intel.daal.algorithms.optimization_solver.objective_function.Result "objective_function.Result" class
 *
 */
public class Batch extends com.intel.daal.algorithms.optimization_solver.sum_of_functions.Batch {
    public Method        method;   /*!< Computation method for the algorithm */
    private Precision    prec;     /*!< Precision of intermediate computations */
    public Parameter parameter;    /*!< Parameters of the algorithm */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the logistic loss objective function algorithm by copying input objects and parameters of another logistic loss objective function algorithm
     * @param context    Context to manage the logistic loss objective function
     * @param other      An algorithm to be used as the source to initialize the input objects
     *                   and parameters of the algorithm
     */
    public Batch(DaalContext context, Batch other) {
        super(context, other.parameter.getNumberOfTerms());
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), method.getValue());
        input = new Input(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new Parameter(getContext(), cGetParameter(this.cObject, prec.getValue(), method.getValue()));
        super.parameter = parameter;
        setPointersToIface();
    }

    /**
     * Constructs the logistic loss objective function algorithm
     *
     * @param context       Context to manage the logistic loss objective function algorithm
     * @param cls           Data type to use in intermediate computations for the logistic loss objective function algorithm, Double.class or Float.class
     * @param method        Logistic loss objective function computation method, @ref Method
     * @param numberOfTerms Number of terms in the logistic loss objective function that can be represented as sum
     */
    public Batch(DaalContext context, Class<? extends Number> cls, Method method, long numberOfTerms) {
        super(context, numberOfTerms);

        this.method = method;

        if (method != Method.defaultDense) {
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

        this.cObject = cInit(prec.getValue(), method.getValue(), numberOfTerms);
        input = new Input(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new Parameter(getContext(), cGetParameter(this.cObject, prec.getValue(), method.getValue()));
        super.parameter = parameter;
        setPointersToIface();
    }

    /**
     * Computes the logistic loss objective function in the batch processing mode
     * @return  Results of the computation
     */
    @Override
    public Result compute() {
        super.compute();
        Result result = new Result(getContext(), cGetResult(cObject));
        return result;
    }

    /**
     * Registers user-allocated memory to store the results of computing the logistic loss objective function
     * in the batch processing mode
     * @param result    Structure to store results of computing the logistic loss objective function
     */
    public void setResult(Result result) {
        cSetResult(cObject, result.getCObject());
    }

    /**
     * Return the input of the algorithm
     * @return Input of the algorithm
     */
    public Input getInput() {
        return (Input) input;
    }

    /**
     * Return the result of the algorithm
     * @return Result of the algorithm
     */
    @Override
    public Result getResult() {
        return new Result(getContext(), cGetResult(cObject));
    }

    /**
     * Returns the newly allocated logistic loss objective function algorithm
     * with a copy of input objects and parameters of this logistic loss objective function algorithm
     * @param context    Context to manage the logistic loss objective function algorithm
     *
     * @return The newly allocated algorithm
     */
    @Override
    public Batch clone(DaalContext context) {
        return new Batch(context, this);
    }

    protected long getCParameter() {
        return cGetParameter(cObject, prec.getValue(), method.getValue());
    }

    protected long getCInput() {
        return cGetInput(cObject, prec.getValue(), method.getValue());
    }

    private native long cInit(int prec, int method, long numberOfTerms);
    private native long cClone(long algAddr, int prec, int method);
    private native long cGetInput(long cObject, int prec, int method);
    private native long cGetParameter(long cObject, int prec, int method);
}
/** @} */
/** @} */
