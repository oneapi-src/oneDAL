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
 * @brief Contains classes for computing the cosine distance
 */
/**
 * @defgroup cosine_distance Cosine Distance Matrix
 * @ingroup analysis
 * @{
 */
/**
 * @defgroup cosine_distance_batch Batch
 * @ingroup cosine_distance
 * @{
 */
package com.intel.daal.algorithms.cosdistance;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.AnalysisBatch;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COSDISTANCE__BATCH"></a>
 * @brief Computes the cosine distance in the batch processing mode.
 * <!-- \n<a href="DAAL-REF-COSDISTANCE-ALGORITHM">Cosine distance algorithm description and usage models</a> -->
 *
 * @par References
 *      - @ref Method class
 *      - @ref InputId class
 *      - @ref ResultId class
 *      - @ref Input class
 *      - @ref Result class
 *
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
     * Constructs the cosine distance algorithm by copying input objects
     * of another cosine distance algorithm
     * @param context    Context to manage the cosine distance algorithm
     * @param other      An algorithm to be used as the source to initialize the input objects
     *                   and parameters of the algorithm
     */
    public Batch(DaalContext context, Batch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;
        this.cObject = cClone(other.cObject, prec.getValue(), this.method.getValue());
        input = new Input(getContext(), cObject, prec, method);
    }

    /**
     * <a name="DAAL-METHOD-ALGORITHM__COSDISTANCE__BATCH__BATCH"></a>
     * Constructs the cosine distance algorithm
     *
     * @param context    Context to manage the cosine distance algorithm
     * @param cls        Data type to use in intermediate computations for cosine distance, Double.class or Float.class
     * @param method     Cosine distance computation method, @ref Method
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
        } else {
            prec = Precision.singlePrecision;
        }

        this.cObject = cInit(prec.getValue(), method.getValue());
        input = new Input(getContext(), cObject, prec, method);
    }

    /**
     * Computes the cosine distance
     * @return  Results of the cosine distance algorithm
     */
    @Override
    public Result compute() {
        super.compute();
        Result result = new Result(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
        return result;
    }

    /**
     * Registers user-allocated memory to store results of the cosine distance algorithm
     * @param result Object to store the results
     */
    public void setResult(Result result) {
        cSetResult(cObject, prec.getValue(), method.getValue(), result.getCObject());
    }

    /**
     * Returns the newly allocated cosine distance algorithm with a copy of input objects
     * of this cosine distance algorithm
     * @param context    Context to manage the cosine distance algorithm
     *
     * @return The newly allocated algorithm
     */
    @Override
    public Batch clone(DaalContext context) {
        return new Batch(context, this);
    }

    private native long cInit(int prec, int method);

    private native long cGetResult(long cAlgorithm, int prec, int method);

    private native void cSetResult(long cAlgorithm, int prec, int method, long cObject);

    private native long cClone(long cAlgorithm, int prec, int method);
}
/** @} */
/** @} */
