/* file: Online.java */
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
 * @defgroup qr_online Online
 * @ingroup qr_without_pivoting
 * @{
 */
package com.intel.daal.algorithms.qr;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.AnalysisOnline;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__QR__ONLINE"></a>
 * @brief Computes the results of the QR decomposition algorithm in the online processing mode
 * <!-- \n<a href="DAAL-REF-QR-ALGORITHM">QR decomposition algorithm description and usage models</a> -->
 *
 * @par References
 *      - InputId class. Identifiers of input objects for the QR decomposition algorithm
 *      - PartialResultId class. Identifiers of partial results of the QR decomposition algorithm
 *      - ResultId class. Results of the QR decomposition algorithm
 */
public class Online extends AnalysisOnline {
    public Input      input; /*!< %Input data */
    public Method     method; /*!< Computation method for the algorithm */
    protected Precision                 prec; /*!< Precision of intermediate computations */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the QR decomposition algorithm by copying input objects and parameters
     * of another QR decomposition algorithm
     * @param context   Context to manage the QR decomposition algorithm
     * @param other     An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    public Online(DaalContext context, Online other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), method.getValue());
        input = new Input(getContext(), cGetInput(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs the QR decomposition algorithm
     * @param context   Context to manage the QR decomposition algorithm
     * @param cls       Data type to use in intermediate computations for the QR decomposition algorithm,
     *                  Double.class or Float.class
     * @param method    computation method, @ref Method
     */
    public Online(DaalContext context, Class<? extends Number> cls, Method method) {
        super(context);

        this.method = method;
        if (this.method != Method.defaultDense) {
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

        this.cObject = InitOnline(prec.getValue(), method.getValue());
        input = new Input(getContext(), cGetInput(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Runs the QR decomposition algorithm computation
     * @return  Partial results of the QR decomposition algorithm in the  online distributed processing mode
     */
    @Override
    public OnlinePartialResult compute() {
        super.compute();
        return new OnlinePartialResult(getContext(), cGetPartialResult(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Computes final results of the QR decomposition algorithm     * @return  Final results of the QR decomposition algorithm
     */
    @Override
    public Result finalizeCompute() {
        super.finalizeCompute();
        return new Result(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Registers user-allocated memory to store the results of the QR decomposition algorithm
     * @param result Structure for storing the results of the QR decomposition algorithm
     */
    public void setResult(Result result) {
        cSetResult(cObject, prec.getValue(), method.getValue(), result.getCObject());
    }

    /**
     * Returns the newly allocated QR decomposition algorithm
     * with a copy of input objects and parameters of this QR decomposition algorithm
     * @param context   Context to manage the QR decomposition algorithm
     *
     * @return The newly allocated algorithm
     */
    @Override
    public Online clone(DaalContext context) {
        return new Online(context, this);
    }

    private native long InitOnline(int prec, int method);

    private native long cGetInput(long algAddr, int prec, int method);

    private native long cGetResult(long algAddr, int prec, int method);

    protected native long cGetPartialResult(long algAddr, int prec, int method);

    private native void cSetResult(long cObject, int prec, int method, long cResult);

    private native long cClone(long algAddr, int prec, int method);
}
/** @} */
