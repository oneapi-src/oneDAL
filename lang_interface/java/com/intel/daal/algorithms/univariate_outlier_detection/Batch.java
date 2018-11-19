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
 * @defgroup univariate_outlier_detection Univariate Outlier Detection
 * @brief Contains classes for computing results of the univariate outlier detection algorithm
 * @ingroup analysis
 * @{
 */
/**
 * @defgroup univariate_outlier_detection_batch Batch
 * @ingroup univariate_outlier_detection
 * @{
 */
/**
 * @brief Contains classes for computing results of the univariate outlier detection algorithm
 */
package com.intel.daal.algorithms.univariate_outlier_detection;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.AnalysisBatch;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__UNIVARIATE_OUTLIER_DETECTION__BATCH"></a>
 * \brief Runs the univariate outlier detection algorithm in the batch processing mode.
 * <!-- \n <a href="DAAL-REF-UNIVARIATE_OUTLIER_DETECTION-ALGORITHM">univariate outlier detection algorithm description and usage models</a> -->
 *
 * \par References
 *      - InputId class. Identifiers of input objects
 *      - ResultId class. Identifiers of results
 */
public class Batch extends AnalysisBatch {
    public Input input; /*!< %Input data */
    public Method method; /*!< Computation method for the algorithm */
    private Result result; /*!< %Result of the algorithm */
    private Precision prec; /*!< Precision of intermediate computations */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the univariate outlier detection algorithm by copying input objects
     * of another univariate outlier detection algorithm
     * @param context   Context to manage created univariate outlier detection algorithm
     * @param other     An algorithm to be used as the source to initialize the input objects
     *                  of the algorithm
     */
    public Batch(DaalContext context, Batch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), method.getValue());
        input = new Input(getContext(), cGetInput(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs the univariate outlier detection algorithm
     * @param context   Context to manage created univariate outlier detection algorithm
     * @param cls       Data type to use in intermediate computations for the univariate outlier detection algorithm,
     *                  Double.class or Float.class
     * @param method    univariate outlier detection computation method, @ref Method
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
        input = new Input(getContext(), cGetInput(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Runs the univariate outlier detection algorithm
     * @return  Univariate outlier detection results
     */
    @Override
    public Result compute() {
        super.compute();
        result = new Result(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
        return result;
    }

    /**
     * Registers user-allocated memory to store univariate outlier detection results
     * @param result    Structure to store univariate outlier detection results
     */
    public void setResult(Result result) {
        cSetResult(cObject, prec.getValue(), method.getValue(), result.getCObject());
    }

    /**
     * Returns the newly allocated univariate outlier detection algorithm
     * with a copy of input objects of this univariate outlier detection algorithm
     * @param context   Context to manage created univariate outlier detection algorithm
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

    private native void cSetResult(long cAlgorithm, int prec, int method, long cResult);

    private native long cClone(long algAddr, int prec, int method);
}
/** @} */
/** @} */
