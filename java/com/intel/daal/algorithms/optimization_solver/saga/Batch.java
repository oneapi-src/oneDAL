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
 * @defgroup SAGA Stochastic Average Gradient Algorithm
 * @brief Contains classes for computing the Adaptive gradient descen
 * @ingroup optimization_solver
 * @{
 */
/**
 * @defgroup saga_batch Batch
 * @ingroup saga
 * @{
 */
package com.intel.daal.algorithms.optimization_solver.saga;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;
import com.intel.daal.algorithms.optimization_solver.iterative_solver.Input;
import com.intel.daal.algorithms.optimization_solver.iterative_solver.Result;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__SAGA__BATCH"></a>
 * @brief %Base interface for the SAGA algorithm in the batch processing mode
 * <!-- \n<a href="DAAL-REF-SAGA-ALGORITHM">SAGA algorithm description and usage models</a> -->
 *
 * @par References
 *      - com.intel.daal.algorithms.optimization_solver.iterative_solver.InputId class
 *      - com.intel.daal.algorithms.optimization_solver.iterative_solver.ResultId class
 *      - com.intel.daal.algorithms.optimization_solver.iterative_solver.Input class
 *
 */
public class Batch extends com.intel.daal.algorithms.optimization_solver.iterative_solver.Batch {

    public Method method; /*!< Computation method for the algorithm */
    private Precision prec; /*!< Precision of intermediate computations */
    public Parameter parameter; /*!< Parameters of the algorithm */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the SAGA algorithm by copying input objects and parameters of another SAGA algorithm
     * @param context    Context to manage the SAGA algorithm
     * @param other      An algorithm to be used as the source to initialize the input objects
     *                   and parameters of the algorithm
     */
    public Batch(DaalContext context, Batch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), method.getValue());
        input = new Input(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new Parameter(getContext(), cGetParameter(this.cObject, prec.getValue(), method.getValue()));
        super.parameter = parameter;
    }

    /**
     * <a name="DAAL-METHOD-ALGORITHMS__OPTIMIZATION_SOLVER__SAGA__BATCH__BATCH"></a>
     * Constructs the SAGA algorithm
     *
     * @param context      Context to manage the SAGA algorithm
     * @param cls          Data type to use in intermediate computations for the SAGA algorithm, Double.class or Float.class
     * @param method       SAGA computation method, @ref Method
     */
    public Batch(DaalContext context, Class<? extends Number> cls, Method method) {
        super(context);

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
        super.parameter = parameter;
    }

    public Batch(DaalContext context, Class<? extends Number> cls, Method method, long cAlgorithm) {
        super(context);

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

        this.cObject = cAlgorithm;
        input = new Input(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new Parameter(getContext(), cGetParameter(this.cObject, prec.getValue(), method.getValue()));
        super.parameter = parameter;
    }

    /**
     * Computes the SAGA in the batch processing mode
     * @return  Results of the computation
     */
    @Override
    public Result compute() {
        super.compute();
        Result result = new Result(getContext(), cGetResult(cObject));
        return result;
    }

    /**
     * Returns the newly allocated SAGA algorithm
     * with a copy of input objects and parameters of this SAGA algorithm
     * @param context    Context to manage the SAGA algorithm
     *
     * @return The newly allocated algorithm
     */
    @Override
    public Batch clone(DaalContext context) {
        return new Batch(context, this);
    }

    private native long cInit(int prec, int method);
    private native long cClone(long algAddr, int prec, int method);
    private native long cGetInput(long cAlgorithm, int prec, int method);
    private native long cGetParameter(long cAlgorithm, int prec, int method);
}
/** @} */
/** @} */
