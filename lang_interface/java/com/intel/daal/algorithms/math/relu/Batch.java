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
 * @defgroup relu Rectifier Linear Unit (ReLU)
 * @brief Contains classes for computing the rectified linear function
 * @ingroup math
 * @{
 */
/**
 * @defgroup relu_batch Batch
 * @ingroup relu
 * @{
 */
package com.intel.daal.algorithms.math.relu;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.AnalysisBatch;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__RELU__BATCH"></a>
 * \brief Computes the rectified linear function in the batch processing mode.
 * <!-- \n<a href="DAAL-REF-RELU-ALGORITHM"> The rectified linear function description and usage models</a> -->
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
     * Constructs the rectified linear function by copying input objects of another rectified linear function
     * @param context    Context to manage the rectified linear function
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
     * <a name="DAAL-METHOD-ALGORITHM__RELU__BATCH__BATCH"></a>
     * Constructs the rectified linear function
     *
     * @param context    Context to manage the rectified linear function
     * @param cls        Data type to use in intermediate computations for the rectified linear function, Double.class or Float.class
     * @param method     The rectified linear function computation method, @ref Method
     */
    public Batch(DaalContext context, Class<? extends Number> cls, Method method) {
        super(context);

        this.method = method;

        if (method != Method.defaultDense && method != Method.fastCSR) {
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
     * Computes the rectified linear function
     * @return  The rectified linear function results
    */
    @Override
    public Result compute() {
        super.compute();
        Result result = new Result(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
        return result;
    }

    /**
     * Registers user-allocated memory to store results of the rectified linear function
     * @param result    Structure to store results of the rectified linear function
     */
    public void setResult(Result result) {
        cSetResult(cObject, prec.getValue(), method.getValue(), result.getCObject());
    }

    /**
     * Returns the newly allocated rectified linear function
     * with a copy of input objects of this rectified linear function
     * @param context    Context to manage the rectified linear function
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
