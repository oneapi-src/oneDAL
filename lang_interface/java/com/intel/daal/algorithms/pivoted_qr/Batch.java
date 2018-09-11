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
 * @defgroup pivoted_qr Pivoted QR Decomposition
 * @brief Contains classes for computing the pivoted QR decomposition
 * @ingroup qr
 * @{
 */
/**
 * @defgroup pivoted_qr_batch Batch
 * @ingroup pivoted_qr
 * @{
 */
/**
 * @brief Contains classes for computing the pivoted QR decomposition
 */
package com.intel.daal.algorithms.pivoted_qr;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.AnalysisBatch;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PIVOTED_QR__BATCH"></a>
 * @brief Computes the results of the pivoted QR algorithm in the batch processing mode
 * <!-- \n<a href="DAAL-REF-PIVOTED_QR-ALGORITHM">Pivoted QR algorithm description and usage models</a> -->
 *
 * @par References
 *      - InputId class. Identifiers of the input objects for the pivoted QR algorithm
 *      - ResultId class. Identifiers of the results of the pivoted QR algorithm
 */
public class Batch extends AnalysisBatch {
    public Input      input;    /*!< %Input data */
    public Parameter  parameter;     /*!< Parameters of the algorithm */
    public Method     method; /*!< Computation method for the algorithm */
    private Precision                 prec; /*!< Precision of intermediate computations */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs pivoted QR algorithm by copying input objects and parameters
     * of another pivoted QR algorithm
     * @param context   Context to manage the Pivoted QR algorithm
     * @param other     An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    public Batch(DaalContext context, Batch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), method.getValue());
        input = new Input(getContext(), cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new Parameter(getContext(), cInitParameter(this.cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs pivoted QR algorithm
     * @param context   Context to manage the Pivoted QR algorithm
     * @param cls       Data type to use in intermediate computations of the  pivoted QR algorithm,
     *                  Double.class or Float.class
     * @param method    Pivoted QR computation method, @ref Method
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
        input = new Input(getContext(), cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new Parameter(getContext(), cInitParameter(this.cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Performs Pivoted QR computation
     * @return  Pivoted QR results
     */
    @Override
    public Result compute() {
        super.compute();
        return new Result(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Registers user-allocated memory to store the results of the pivoted QR algorithm
     * @param result Structure for storing the results of the pivoted QR algorithm
     */
    public void setResult(Result result) {
        cSetResult(cObject, prec.getValue(), method.getValue(), result.getCObject());
    }

    /**
     * Returns the newly allocated pivoted QR algorithm
     * with a copy of input objects and parameters of this pivoted QR algorithm
     * @param context   Context to manage the Pivoted QR algorithm
     *
     * @return The newly allocated algorithm
     */
    @Override
    public Batch clone(DaalContext context) {
        return new Batch(context, this);
    }

    private native long cInit(int prec, int method);

    private native long cInitParameter(long aldAddr, int prec, int method);

    private native long cGetInput(long aldAddr, int prec, int method);

    private native long cGetResult(long aldAddr, int prec, int method);

    private native void cSetResult(long cObject, int prec, int method, long cResult);

    private native long cClone(long aldAddr, int prec, int method);
}
/** @} */
/** @} */
