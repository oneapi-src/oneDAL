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
 * @defgroup logistic Logistic
 * @brief Contains classes for computing the logistic function
 * @ingroup math
 * @{
 */
/**
 * @defgroup logistic_batch Batch
 * @ingroup logistic
 * @{
 */
package com.intel.daal.algorithms.math.logistic;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.AnalysisBatch;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LOGISTIC__BATCH"></a>
 * \brief Computes logistic function in the batch processing mode.
 * <!-- \n<a href="DAAL-REF-LOGISTIC-ALGORITHM">Logistic function description and usage models</a> -->
 *
 * \par References
 *      - @ref InputId class
 *      - @ref ResultId class
 *
 */
public class Batch extends AnalysisBatch {
    public Input      input;    /*!< %Input data */
    public Method     method;   /*!< Computation method for the function */
    private Precision prec;     /*!< Data type to use in intermediate computations for the function */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs logistic function by copying input objects of another logistic function
     * @param context    Context to manage the logistic function
     * @param other      An function to be used as the source to initialize the input objects of the function
     */
    public Batch(DaalContext context, Batch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), method.getValue());
        input = new Input(context, cGetInput(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * <a name="DAAL-METHOD-ALGORITHM__LOGISTIC__BATCH__BATCH"></a>
     * Constructs the logistic function
     *
     * @param context    Context to manage the logistic function
     * @param cls        Data type to use in intermediate computations for the logistic function, Double.class or Float.class
     * @param method     The logistic function computation method, @ref Method
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
    }

    /**
     * Computes the logistic function
     * @return  The logistic function result
    */
    @Override
    public Result compute() {
        super.compute();
        Result result = new Result(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
        return result;
    }

    /**
     * Registers user-allocated memory to store the result of the logistic function
     * @param result    Structure to store the result of the logistic function
     */
    public void setResult(Result result) {
        cSetResult(cObject, prec.getValue(), method.getValue(), result.getCObject());
    }

    /**
     * Returns the newly allocated logistic function
     * with a copy of the input object of this logistic function
     * @param context    Context to manage the logistic function
     *
     * @return The newly allocated function
     */
    @Override
    public Batch clone(DaalContext context) {
        return new Batch(context, this);
    }

    private native long cInit(int prec, int method);
    private native void cSetResult(long cAlgorithm, int prec, int method, long cObject);
    private native long cGetResult(long cAlgorithm, int prec, int method);
    private native long cGetInput(long cAlgorithm, int prec, int method);
    private native long cClone(long algAddr, int prec, int method);
}
/** @} */
/** @} */
