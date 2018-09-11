/* file: DistributedStep1Local.java */
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
 * @ingroup neural_networks_training_distributed
 * @{
 */
package com.intel.daal.algorithms.neural_networks.training;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.AnalysisDistributed;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.ComputeStep;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;
import com.intel.daal.algorithms.optimization_solver.iterative_solver.Batch;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__TRAINING__DISTRIBUTEDSTEP1LOCAL"></a>
 * @brief Provides methods for neural network model-based training in the distributed processing mode
 */
public class DistributedStep1Local extends AnalysisDistributed {
    public DistributedStep1LocalInput input;         /*!< %Input data */
    public TrainingParameter          parameter;     /*!< Parameters of the algorithm */
    private TrainingMethod            method;        /*!< Computation method for the algorithm */
    private Precision                 prec;          /*!< Precision of intermediate computations */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the neural network training algorithm by copying input objects and parameters
     * of another algorithm
     * @param context   Context to manage the algorithm
     * @param other     An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    public DistributedStep1Local(DaalContext context, DistributedStep1Local other) {
        super(context);
        method = other.method;
        prec = other.prec;
        cObject = cClone(other.cObject, prec.getValue(), method.getValue());
        input     = new DistributedStep1LocalInput(getContext(), cGetInput (cObject, prec.getValue(), method.getValue()));
        parameter = new TrainingParameter(getContext(), cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs the neural network training algorithm
     * @param context Context to manage the algorithm
     * @param cls       Data type to use in intermediate computations for the algorithm,
     *                  Double.class or Float.class
     * @param method    Computation method of the algorithm, @ref TrainingMethod
     */
    public DistributedStep1Local(DaalContext context, Class<? extends Number> cls, TrainingMethod method) {
        super(context);
        initialize(context, Float.class, TrainingMethod.defaultDense);
    }

    /**
     * Constructs neural network model-based training object with Float data type used for intermediate computations and default computation method
     * @param context   Context to manage the neural network training object
     */
    public DistributedStep1Local(DaalContext context) {
        super(context);
        initialize(context, Float.class, TrainingMethod.defaultDense);
    }

    private void initialize(DaalContext context, Class<? extends Number> cls, TrainingMethod method) {
        this.method = method;

        if (method != TrainingMethod.defaultDense && method != TrainingMethod.feedforwardDense) {
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

        cObject = cInit(prec.getValue(), method.getValue());
        input = new DistributedStep1LocalInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new TrainingParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Runs the step 1 of the neural network training algorithm
     * @return  Partial results of the neural network training algorithm
     */
    @Override
    public PartialResult compute() {
        super.compute();
        return new PartialResult(getContext(), cGetPartialResult(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Computes the results of the neural network training algorithm
     * @return  Results of the neural network training algorithm
     */
    @Override
    public TrainingResult finalizeCompute() {
        super.finalizeCompute();
        return new TrainingResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Registers user-allocated memory to store partial results of the neural network training algorithm
     * @param partialResult         Structure to store partial results of the neural network training algorithm
     */
    public void setPartialResult(PartialResult partialResult) {
        cSetPartialResult(cObject, prec.getValue(), method.getValue(), partialResult.getCObject());
    }

    /**
     * Registers user-allocated memory to store the results of the neural network training algorithm
     * @param result    Structure to store the results of the neural network training algorithm
     */
    public void setResult(TrainingResult result) {
        cSetResult(cObject, prec.getValue(), method.getValue(), result.getCObject());
    }

    /**
     * Returns the newly allocated neural network training algorithm with a copy of input objects
     * and parameters of this neural network training algorithm
     * @param context   Context to manage the algorithm
     *
     * @return The newly allocated algorithm
     */
    @Override
    public DistributedStep1Local clone(DaalContext context) {
        return new DistributedStep1Local(context, this);
    }

    private native long cInit(int prec, int method);
    private native long cInitParameter(long addr, int prec, int method);
    private native long cGetInput(long addr, int prec, int method);
    private native long cGetResult(long addr, int prec, int method);
    private native void cSetResult(long addr, int prec, int method, long cResult);
    private native long cGetPartialResult(long addr, int prec, int method);
    private native void cSetPartialResult(long addr, int prec, int method, long cPartialResult);
    private native long cClone(long addr, int prec, int method);
}
/** @} */
