/* file: InitDistributedStep1Local.java */
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
 * @defgroup dbscan_distributed Distributed
 * @ingroup dbscan
 * @{
 */
/**
 * @brief Contains classes of the DBSCAN algorithm
 */
package com.intel.daal.algorithms.gbt.regression.init;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.AnalysisDistributed;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__InitDistributedStep1Local"></a>
 * @brief Runs the DBSCAN algorithm in the first step of the distributed processing mode
 */
public class InitDistributedStep1Local extends AnalysisDistributed {
    public  InitDistributedStep1LocalInput input;      /*!< %Input data */
    public  InitParameter                  parameter;  /*!< Parameters of the algorithm */
    public  InitMethod                     method;     /*!< Computation method for the algorithm */
    private Precision                      precision;  /*!< Precision of intermediate computations */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the DBSCAN algorithm by copying input objects and parameters
     * of another DBSCAN algorithm
     * @param context   Context to manage the algorithm
     * @param other     An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    public InitDistributedStep1Local(DaalContext context, InitDistributedStep1Local other) {
        super(context);
        this.method = other.method;
        precision = other.precision;

        this.cObject = cClone(other.cObject, precision.getValue(), this.method.getValue());

        input     = new InitDistributedStep1LocalInput(getContext(), cGetInput     (cObject, precision.getValue(), method.getValue()));
        parameter = new InitParameter                 (getContext(), cInitParameter(cObject, precision.getValue(), method.getValue()));
    }

    /**
     * Constructs the DBSCAN algorithm
     * @param context    Context to manage the algorithm
     * @param cls        Data type to use in intermediate computations for the algorithm,
     *                   Double.class or Float.class
     * @param method     Computation method of the algorithm, @ref Method
     * @param maxBins
     */
    public InitDistributedStep1Local(DaalContext context, Class<? extends Number> cls, InitMethod method, long maxBins) {
        super(context);

        this.method = method;
        if (cls != Double.class && cls != Float.class) {
            throw new IllegalArgumentException("type unsupported");
        }

        if (this.method != InitMethod.defaultDense) {
            throw new IllegalArgumentException("method unsupported");
        }

        if (cls == Double.class) {
            precision = Precision.doublePrecision;
        } else {
            precision = Precision.singlePrecision;
        }

        this.cObject = cInit(precision.getValue(), this.method.getValue(), maxBins);

        input     = new InitDistributedStep1LocalInput(getContext(), cGetInput     (cObject, precision.getValue(), method.getValue()));
        parameter = new InitParameter                 (getContext(), cInitParameter(cObject, precision.getValue(), method.getValue()));
        parameter.setMaxBins(maxBins);
    }

    /**
     * Runs the DBSCAN algorithm
     * @return  Partial results of the DBSCAN algorithm
     */
    @Override
    public InitDistributedPartialResultStep1 compute() {
        super.compute();
        return new InitDistributedPartialResultStep1(getContext(), cGetPartialResult(cObject, precision.getValue(), method.getValue()));
    }

    /**
     * Registers user-allocated memory to store partial results of the DBSCAN algorithm
     * @param partialResult         Structure to store partial results of the DBSCAN algorithm
     */
    public void setPartialResult(InitDistributedPartialResultStep1 partialResult) {
        cSetPartialResult(cObject, precision.getValue(), method.getValue(), partialResult.getCObject());
    }

    /**
     * Returns the newly allocated DBSCAN algorithm with a copy of input objects
     * and parameters of this DBSCAN algorithm
     * @param context   Context to manage the algorithm
     *
     * @return The newly allocated algorithm
     */
    @Override
    public InitDistributedStep1Local clone(DaalContext context) {
        return new InitDistributedStep1Local(context, this);
    }

    private native long cInit(int precision, int method, long maxBins);
    private native long cInitParameter(long addr, int precision, int method);
    private native long cGetInput(long addr, int precision, int method);
    private native long cGetPartialResult(long addr, int precision, int method);
    private native void cSetPartialResult(long addr, int precision, int method, long cPartialResult);
    private native long cClone(long addr, int precision, int method);
}
/** @} */
