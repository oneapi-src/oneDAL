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
 * @defgroup minmax Min-max
 * @brief Contains classes for computing Min-max normalization algorithms
 * @ingroup normalization
 * @{
 */
/**
 * @defgroup minmax_batch Batch
 * @ingroup minmax
 * @{
 */
/**
 * @brief Contains classes for computing Min-max normalization solvers
 */
package com.intel.daal.algorithms.normalization.minmax;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.AnalysisBatch;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;
import com.intel.daal.algorithms.ComputeMode;

/**
 * <a name="DAAL-CLASS-ALGORITHMS-ALGORITHMS__NORMALIZATION__MINMAX__BATCH"></a>
 * \brief Computes Min-max normalization in the batch processing mode.
 * <!-- \n<a href="DAAL-REF-MINMAX-ALGORITHM">Min-max normalization algorithm description and usage models</a> -->
 *
 * \par References
 *      - @ref InputId class
 *      - @ref ResultId class
 *
 */
public class Batch extends AnalysisBatch {
    public Input      input;     /*!< %Input data */
    public Method     method;    /*!< Computation method for the algorithm */
    private Precision prec;      /*!< Precision of computations */
    public Parameter  parameter; /*!< Parameters of the algorithm */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * <a name="DAAL-CLASS-ALGORITHMS-ALGORITHMS__NORMALIZATION__MINMAX__BATCH__BATCH"></a>
     * Constructs the Min-max normalization algorithm
     *
     * @param context    Context to manage the Min-max normalization algorithm
     * @param cls        Data type to use in intermediate computations for Min-max normalization, Double.class or Float.class
     * @param method     Min-max normalization computation method, @ref Method
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
        parameter = new Parameter(context, cGetParameter(cObject, prec.getValue(), method.getValue()));
    }

    /**
    * Constructs algorithm that computes normalization by copying input objects and parameters
    * of another algorithm
    * @param context      Context to manage the normalization algorithms
    * @param other        An algorithm to be used as the source to initialize the input objects
    *                     and parameters of the algorithm
    */
    public Batch(DaalContext context, Batch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), this.method.getValue());
        input = new Input(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new Parameter(getContext(), cGetParameter(this.cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Computes Min-max normalization
     * @return  Min-max normalization results
    */
    @Override
    public Result compute() {
        super.compute();
        Result result = new Result(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
        return result;
    }

    /**
     * Registers user-allocated memory to store the result of the Min-max normalization algorithm
     * @param result    Structure to store the result of the Min-max normalization algorithm
     */
    public void setResult(Result result) {
        cSetResult(cObject, prec.getValue(), method.getValue(), result.getCObject());
    }

    /**
     * Returns the newly allocated algorithm that computes normalization
     * with a copy of input objects and parameters of this algorithm
     * @param context      Context to manage the normalization algorithms
     *
     * @return The newly allocated algorithm
     */
    @Override
    public Batch clone(DaalContext context) {
        return new Batch(context, this);
    }

    private native long cInit(int prec, int method);
    private native long cGetParameter(long cAlgorithm, int prec, int method);
    private native long cGetInput(long cAlgorithm, int prec, int method);
    private native long cGetResult(long cAlgorithm, int prec, int method);
    private native void cSetResult(long cAlgorithm, int prec, int method, long cObject);
    private native long cClone(long algAddr, int prec, int method);
}
/** @} */
/** @} */
