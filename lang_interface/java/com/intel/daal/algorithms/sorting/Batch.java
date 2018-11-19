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
 * @defgroup sorting Sorting
 * @brief Contains classes to run the sorting algorithms
 * @ingroup analysis
 * @{
 */
/**
 * @defgroup sorting_batch Batch
 * @ingroup sorting
 * @{
 */
/**
 * @brief Contains classes to run the sorting
 */
package com.intel.daal.algorithms.sorting;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.AnalysisBatch;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__SORTING__BATCH"></a>
 * @brief Sorts data in the batch processing mode.
 * <!-- \n<a href="DAAL-REF-SORTING-ALGORITHM">Sorting algorithm description and usage models</a> -->
 *
 * @tparam algorithmFPType  Data type to use in intermediate computations for the sorting, double or float
 * @tparam method           Sorting computation method, @ref daal::algorithms::sorting::Method
 *
 * @par Enumerations
 *      - @ref Method   Sorting computation methods
 *      - @ref InputId  Identifiers of sorting input objects
 *      - @ref ResultId Identifiers of sorting results
 */
public class Batch extends AnalysisBatch {
    public Input          input;     /*!< %Input data */
    public Method     method; /*!< Computation method for the algorithm */
    private Precision                 prec; /*!< Precision of intermediate computations */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs sorting algorithm by copying input objects and parameters
     * of another sorting algorithm
     * @param context      Context to manage the sorting
     * @param other        An algorithm to be used as the source to initialize the input objects
     *                     and parameters of the algorithm
     */
    public Batch(DaalContext context, Batch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), this.method.getValue());
        input = new Input(getContext(), cGetInput(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * <a name="DAAL-METHOD-ALGORITHM__SORTING__BATCH__BATCH"></a>
     *  @brief Sorts data in the batch processing mode
     * @param context      Context to manage the sorting
     * @param cls          Data type to use in intermediate computations for the sorting algorithms, Double.class or Float.class
     * @param method       Sorting computation methods, @ref Method
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
        }
        else {
            prec = Precision.singlePrecision;
        }

        this.cObject = cInit(prec.getValue(), this.method.getValue());

        input = new Input(getContext(), cGetInput(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Runs the sorting algorithm
     * @return  Sorting computation results
     */
    @Override
    public Result compute() {
        super.compute();
        Result result = new Result(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
        return result;
    }

    /**
     * Registers user-allocated memory to store results of the sorting
     * @param result    Structure to store results of the sorting
     */
    public void setResult(Result result) {
        cSetResult(cObject, prec.getValue(), method.getValue(), result.getCObject());
    }

    /**
     * Returns the newly allocated sorting algorithm
     * with a copy of input objects and parameters of this sorting algorithm
     * @param context      Context to manage the sorting
     *
     * @return The newly allocated algorithm
     */
    @Override
    public Batch clone(DaalContext context) {
        return new Batch(context, this);
    }

    private native long cInit(int prec, int method);

    private native long cGetInput(long algAddr, int prec, int method);

    private native long cGetResult(long algAddr, int prec, int method);

    private native void cSetResult(long cObject, int prec, int method, long cResult);

    private native long cClone(long algAddr, int prec, int method);
}
/** @} */
/** @} */
