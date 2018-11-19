/* file: DistributedStep2Master.java */
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
 * @ingroup implicit_als_training_distributed
 * @{
 */
package com.intel.daal.algorithms.implicit_als.training;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.ComputeStep;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.TrainingDistributed;
import com.intel.daal.algorithms.implicit_als.Parameter;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__DISTRIBUTEDSTEP2MASTER"></a>
 * @brief Runs the implicit ALS training algorithm in the second step of the distributed processing mode
 */
public class DistributedStep2Master extends TrainingDistributed {
    public DistributedStep2MasterInput    input;        /*!< %Input data */
    public Parameter  parameter;     /*!< Parameters of the algorithm */
    public TrainingMethod method;   /*!< %Training method for the algorithm */
    private Precision                 prec; /*!< Precision of intermediate computations */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the implicit ALS training algorithm in the second step of the distributed processing mode
     * by copying input objects and parameters of another implicit ALS training algorithm
     * @param context   Context to manage the implicit ALS training algorithm
     * @param other     An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    public DistributedStep2Master(DaalContext context, DistributedStep2Master other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), this.method.getValue());
        input = new DistributedStep2MasterInput(getContext(), cObject, prec, method);
        parameter = new Parameter(getContext(), cInitParameter(this.cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs the implicit ALS training algorithm in the second step of the distributed processing mode
     * @param context   Context to manage the implicit ALS training algorithm
     * @param cls       Data type to use in intermediate computations for the implicit ALS training algorithm,
     *                  Double.class or Float.class
     * @param method    Implicit ALS computation method, @ref TrainingMethod
     */
    public DistributedStep2Master(DaalContext context, Class<? extends Number> cls, TrainingMethod method) {
        super(context);

        this.method = method;
        if (this.method != TrainingMethod.fastCSR) {
            throw new IllegalArgumentException("method unsupported");
        }

        if (cls == Double.class) {
            prec = Precision.doublePrecision;
        } else if (cls == Float.class) {
            prec = Precision.singlePrecision;
        } else {
            throw new IllegalArgumentException("type unsupported");
        }

        this.cObject = cInit(prec.getValue(), this.method.getValue());

        input = new DistributedStep2MasterInput(getContext(), cObject, prec, method);
        parameter = new Parameter(getContext(), cInitParameter(this.cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Computes partial results of the implicit ALS training algorithm obtained in the second step of the distributed processing mode
     * @return  Partial results of the implicit ALS training algorithm obtained in the second step of the distributed processing mode
     */
    @Override
    public DistributedPartialResultStep2 compute() {
        super.compute();
        return new DistributedPartialResultStep2(getContext(), cGetPartialResult(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Registers user-allocated memory to store partial results of the implicit ALS training algorithm obtained
     * in the second step of the distributed processing mode
     * @param partialResult         Structure to store partial results of the implicit ALS training algorithm
     * obtained in the second step of the distributed processing mode
     */
    public void setPartialResult(DistributedPartialResultStep2 partialResult) {
        cSetPartialResult(this.cObject, prec.getValue(), method.getValue(), partialResult.getCObject());
    }

    /**
     * Returns the newly allocated ALS training algorithm in the second step of the distributed processing mode
     * with a copy of input objects and parameters of this ALS training algorithm
     * @param context   Context to manage the implicit ALS training algorithm
     *
     * @return The newly allocated algorithm
     */
    @Override
    public DistributedStep2Master clone(DaalContext context) {
        return new DistributedStep2Master(context, this);
    }

    private native long cInit(int prec, int method);

    private native long cInitParameter(long algAddr, int prec, int method);

    private native long cGetPartialResult(long cObject, int prec, int method);

    private native void cSetPartialResult(long cObject, int prec, int method, long cPartialResult);

    private native long cClone(long algAddr, int prec, int method);
}
/** @} */
