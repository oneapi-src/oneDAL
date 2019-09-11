/* file: Batch.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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
 * @defgroup optimization_solver Optimization Solvers
 * @brief Contains classes for optimization solver algorithms
 * @ingroup analysis
 */
/**
 * @defgroup lbfgs Limited-Memory-Broyden-Fletcher-Goldfarb-Shanno Algorithm
 * @brief Contains classes for computing the limited memory Broyden-Fletcher-Goldfarb-Shanno algorithm
 * @ingroup optimization_solver
 * @{
 */
/**
 * @defgroup lbfgs_batch Batch
 * @ingroup lbfgs
 * @{
 */
/**
 * @brief Contains classes for computing limited memory Broyden-Fletcher-Goldfarb-Shanno algorithm
 */
package com.intel.daal.algorithms.optimization_solver.lbfgs;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.AnalysisBatch;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;
import com.intel.daal.algorithms.optimization_solver.iterative_solver.Input;
import com.intel.daal.algorithms.optimization_solver.iterative_solver.Result;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__LBFGS__BATCH"></a>
 * @brief Computes the results of LBFGS algorithm in the batch processing mode
 * <!-- \n<a href="DAAL-REF-LBFGS-ALGORITHM">LBFGS algorithm description and usage models</a> -->
 *
 * @par References
 *      - @ref com.intel.daal.algorithms.optimization_solver.iterative_solver.InputId class
 *      - @ref com.intel.daal.algorithms.optimization_solver.iterative_solver.Input class
 *      - @ref com.intel.daal.algorithms.optimization_solver.iterative_solver.ResultId class
 *
 */
public class Batch extends com.intel.daal.algorithms.optimization_solver.iterative_solver.Batch {

    public Method method;  /*!< Computation method for the algorithm */
    public Parameter parameter;     /*!< Parameters of the algorithm */
    private Precision prec; /*!< Precision of intermediate computations */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the LBFGS algorithm by copying input objects and parameters of another LBFGS algorithm
     * @param context    Context to manage the LBFGS algorithm
     * @param other      An algorithm to be used as the source to initialize the input objects
     *                   and parameters of this algorithm
     */
    public Batch(DaalContext context, Batch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), method.getValue());
        input = new Input(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new Parameter(context, cGetParameter(this.cObject, prec.getValue(), method.getValue()));
        super.parameter = parameter;
    }

    /**
     * Constructs the LBFGS algorithm
     *
     * @param context      Context to manage the LBFGS algorithm
     * @param cls          Data type to use in intermediate computations for the LBFGS algorithm, Double.class or Float.class
     * @param method       LBFGS computation method, @ref Method
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
        parameter = new Parameter(context, cGetParameter(this.cObject, prec.getValue(), method.getValue()));
        super.parameter = parameter;
    }

    /**
     * Computes the results of LBFGS algorithm in the batch processing mode
     * @return  Results of the computation
     */
    @Override
    public Result compute() {
        super.compute();
        return new Result(getContext(), cGetResult(cObject));
    }

    /**
     * Returns the newly allocated LBFGS algorithm
     * with a copy of input objects and parameters of this LBFGS algorithm
     * @param context    Context to manage the LBFGS algorithm
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
