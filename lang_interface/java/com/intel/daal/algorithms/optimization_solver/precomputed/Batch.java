/* file: Batch.java */
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
 * @defgroup precomputed Objective function with precomputed characteristics
 * @brief Contains classes for computing the Objective function with precomputed characteristics
 * @ingroup objective_function
 * @{
 */
/**
 * @defgroup precomputed_batch Batch
 * @ingroup precomputed
 * @{
 */
package com.intel.daal.algorithms.optimization_solver.precomputed;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.AnalysisBatch;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;
import com.intel.daal.algorithms.optimization_solver.objective_function.Result;
import com.intel.daal.algorithms.optimization_solver.sum_of_functions.Input;
import com.intel.daal.algorithms.optimization_solver.sum_of_functions.Parameter;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__PRECOMPUTED__BATCH"></a>
 * @brief Computes the objective function with precomputed characteristics in the batch processing mode
 * <!-- \n<a href="DAAL-REF-PRECOMPUTED-ALGORITHM">The objective function with precomputed characteristics algorithm description and usage models</a> -->
 *
 * @par References
 *      - \ref com.intel.daal.algorithms.optimization_solver.objective_function.ResultId "objective_function.ResultId" class
 *      - \ref com.intel.daal.algorithms.optimization_solver.objective_function.Result "objective_function.Result" class
 *
 */
public class Batch extends com.intel.daal.algorithms.optimization_solver.sum_of_functions.Batch {
    public Method method; /*!< Computation method for the algorithm */
    private Precision                 prec; /*!< Precision of intermediate computations */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the objective function with precomputed characteristics algorithm by copying input objects
     * and parameters of another objective function with precomputed characteristics algorithm
     * @param context    Context to manage the objective function with precomputed characteristics algorithm
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
        setPointersToIface();
    }

    /**
     * Constructs the objective function with precomputed characteristics algorithm
     *
     * @param context       Context to manage the objective function with precomputed characteristics algorithm
     * @param cls           Data type to use in intermediate computations for the objective function with precomputed characteristics algorithm, Double.class or Float.class
     * @param method        Objective function with precomputed characteristics computation method, @ref Method
     */
    public Batch(DaalContext context, Class<? extends Number> cls, Method method) {
        super(context, 1);

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

        this.cObject = cInit(prec.getValue(), method.getValue());
        input = new Input(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new Parameter(getContext(), cGetParameter(this.cObject, prec.getValue(), method.getValue()));
        setPointersToIface();
    }

    /**
     * Computes the objective function with precomputed characteristics in the batch processing mode
     * @return  Results of the computation
     */
    @Override
    public Result compute() {
        super.compute();
        Result result = new Result(getContext(), cGetResult(cObject));
        return result;
    }

    /**
     * Returns the newly allocated objective function with precomputed characteristics algorithm
     * with a copy of input objects and parameters of this objective function with precomputed characteristics algorithm
     * @param context    Context to manage the objective function with precomputed characteristics algorithm
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

    private native long cInit(int prec, int method);
    private native long cClone(long algAddr, int prec, int method);
    private native long cGetInput(long cObject, int prec, int method);
    private native long cGetParameter(long cObject, int prec, int method);
}
/** @} */
/** @} */
