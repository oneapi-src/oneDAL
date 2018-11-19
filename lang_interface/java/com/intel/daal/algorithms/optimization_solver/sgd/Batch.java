/* file: Batch.java */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/**
 * @defgroup sgd Stochastic Gradient Descent Algorithm
 * @brief Contains classes for computing the Stochastic gradient descent
 * @ingroup optimization_solver
 * @{
 */
/**
 * @defgroup sgd_batch Batch
 * @ingroup sgd
 * @{
 */
package com.intel.daal.algorithms.optimization_solver.sgd;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.AnalysisBatch;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;
import com.intel.daal.algorithms.optimization_solver.sgd.*;
import com.intel.daal.algorithms.optimization_solver.iterative_solver.Input;
import com.intel.daal.algorithms.optimization_solver.iterative_solver.Result;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__SGD__BATCH"></a>
 * @brief %Base interface for the SGD algorithm in the batch processing mode
 * <!-- \n<a href="DAAL-REF-SGD-ALGORITHM">SGD algorithm description and usage models</a> -->
 *
 * @par References
 *      - Parameter class
 *      - com.intel.daal.algorithms.optimization_solver.iterative_solver.InputId class
 *      - com.intel.daal.algorithms.optimization_solver.iterative_solver.ResultId class
 *      - com.intel.daal.algorithms.optimization_solver.iterative_solver.Input class
 *
 */
public class Batch extends com.intel.daal.algorithms.optimization_solver.iterative_solver.Batch {

    public Method method; /*!< Computation method for the algorithm */
    private Precision prec; /*!< Precision of intermediate computations */
    public BaseParameter parameter; /*!< Parameters of the algorithm */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the SGD algorithm by copying input objects and parameters of another SGD algorithm
     * @param context    Context to manage the SGD algorithm
     * @param other      An algorithm to be used as the source to initialize the input objects
     *                   and parameters of the algorithm
     */
    public Batch(DaalContext context, Batch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), method.getValue());
        input = new Input(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        if(method == Method.defaultDense) {
            parameter = new ParameterDefaultDense(getContext(), cGetParameter(this.cObject, prec.getValue(), method.getValue()));
        }
        else if(method == Method.miniBatch) {
            parameter = new ParameterMiniBatch(getContext(), cGetParameter(this.cObject, prec.getValue(), method.getValue()));
        }
        else if(method == Method.momentum) {
            parameter = new ParameterMomentum(getContext(), cGetParameter(this.cObject, prec.getValue(), method.getValue()));
        }
        super.parameter = parameter;
    }

    /**
     * <a name="DAAL-METHOD-ALGORITHMS__OPTIMIZATION_SOLVER__SGD__BATCH__BATCH"></a>
     * Constructs the SGD algorithm
     *
     * @param context      Context to manage the SGD algorithm
     * @param cls          Data type to use in intermediate computations for the SGD algorithm, Double.class or Float.class
     * @param method       SGD computation method, @ref Method
     */
    public Batch(DaalContext context, Class<? extends Number> cls, Method method) {
        super(context);

        this.method = method;

        if (method != Method.defaultDense && method != Method.miniBatch && method != Method.momentum) {
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
        if(method == Method.defaultDense) {
            parameter = new ParameterDefaultDense(getContext(), cGetParameter(this.cObject, prec.getValue(), method.getValue()));
        }
        else if(method == Method.miniBatch) {
            parameter = new ParameterMiniBatch(getContext(), cGetParameter(this.cObject, prec.getValue(), method.getValue()));
        }
        else if(method == Method.momentum) {
            parameter = new ParameterMomentum(getContext(), cGetParameter(this.cObject, prec.getValue(), method.getValue()));
        }
        super.parameter = parameter;
    }

    public Batch(DaalContext context, Class<? extends Number> cls, Method method, long cAlgorithm) {
        super(context);

        this.method = method;

        if (method != Method.defaultDense && method != Method.miniBatch && method != Method.momentum) {
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
        if(method == Method.defaultDense) {
            parameter = new ParameterDefaultDense(getContext(), cGetParameter(this.cObject, prec.getValue(), method.getValue()));
        }
        else if(method == Method.miniBatch) {
            parameter = new ParameterMiniBatch(getContext(), cGetParameter(this.cObject, prec.getValue(), method.getValue()));
        }
        else if(method == Method.momentum) {
            parameter = new ParameterMomentum(getContext(), cGetParameter(this.cObject, prec.getValue(), method.getValue()));
        }
        super.parameter = parameter;
    }

    /**
     * Computes the SGD in the batch processing mode
     * @return  Results of the computation
     */
    @Override
    public Result compute() {
        super.compute();
        Result result = new Result(getContext(), cGetResult(cObject));
        return result;
    }

    /**
     * Returns the newly allocated SGD algorithm
     * with a copy of input objects and parameters of this SGD algorithm
     * @param context    Context to manage the SGD algorithm
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
