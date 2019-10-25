/* file: DistributedStep3Local.java */
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
 * @defgroup gbt_distributed
 * @{
 */
package com.intel.daal.algorithms.gbt.regression.training;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.AnalysisDistributed;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__REGRESSION__TRAINING__DISTRIBUTEDSTEP3LOCAL"></a>
 * @brief Provides methods for model-based training in the third step of distributed processing mode
 */
public class DistributedStep3Local extends AnalysisDistributed {
    public  DistributedStep3LocalInput input;      /*!< %Input data */
    public  Parameter                  parameter;  /*!< Parameters of the algorithm */
    public  Method                     method;     /*!< Computation method for the algorithm */
    private Precision                  precision;  /*!< Precision of intermediate computations */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs a gradient boosted trees training algorithm by copying input objects and parameters
     * of another gradient boosted trees training algorithm in the third step of distributed processing mode
     * @param context   Context to manage the algorithm
     * @param other     An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    public DistributedStep3Local(DaalContext context, DistributedStep3Local other) {
        super(context);
        this.method = other.method;
        precision = other.precision;

        this.cObject = cClone(other.cObject, precision.getValue(), this.method.getValue());

        input     = new DistributedStep3LocalInput(getContext(), cGetInput     (cObject, precision.getValue(), method.getValue()));
        parameter = new Parameter                 (getContext(), cInitParameter(cObject, precision.getValue(), method.getValue()));
    }

    /**
     * Constructs a gradient boosted trees training algorithm
     * @param context    Context to manage the algorithm
     * @param cls        Data type to use in intermediate computations for the algorithm,
     *                   Double.class or Float.class
     * @param method     Computation method of the algorithm, @ref Method
     */
    public DistributedStep3Local(DaalContext context, Class<? extends Number> cls, Method method) {
        super(context);

        this.method = method;
        if (cls != Double.class && cls != Float.class) {
            throw new IllegalArgumentException("type unsupported");
        }

        if (this.method != Method.defaultDense) {
            throw new IllegalArgumentException("method unsupported");
        }

        if (cls == Double.class) {
            precision = Precision.doublePrecision;
        } else {
            precision = Precision.singlePrecision;
        }

        this.cObject = cInit(precision.getValue(), this.method.getValue());

        input     = new DistributedStep3LocalInput(getContext(), cGetInput     (cObject, precision.getValue(), method.getValue()));
        parameter = new Parameter                 (getContext(), cInitParameter(cObject, precision.getValue(), method.getValue()));
    }

    /**
     * Runs the a gradient boosted trees training algorithm
     * @return  Partial results of the a gradient boosted trees training algorithm
     */
    @Override
    public DistributedPartialResultStep3 compute() {
        super.compute();
        return new DistributedPartialResultStep3(getContext(), cGetPartialResult(cObject, precision.getValue(), method.getValue()));
    }

    /**
     * Registers user-allocated memory to store partial results of the a gradient boosted trees training algorithm
     * @param partialResult         Structure to store partial results of the a gradient boosted trees training algorithm
     */
    public void setPartialResult(DistributedPartialResultStep3 partialResult) {
        cSetPartialResult(cObject, precision.getValue(), method.getValue(), partialResult.getCObject());
    }

    /**
     * Returns the newly allocated a gradient boosted trees training algorithm with a copy of input objects
     * and parameters of this a gradient boosted trees training algorithm
     * @param context   Context to manage the algorithm
     *
     * @return The newly allocated algorithm
     */
    @Override
    public DistributedStep3Local clone(DaalContext context) {
        return new DistributedStep3Local(context, this);
    }

    private native long cInit(int precision, int method);
    private native long cInitParameter(long addr, int precision, int method);
    private native long cGetInput(long addr, int precision, int method);
    private native long cGetPartialResult(long addr, int precision, int method);
    private native void cSetPartialResult(long addr, int precision, int method, long cPartialResult);
    private native long cClone(long addr, int precision, int method);
}
/** @} */
