/* file: DistributedStep2Master.java */
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
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__TRAINING__DISTRIBUTEDSTEP2MASTER"></a>
 * @brief Computes neural network training in the distributed processing mode on the master node
 */
public class DistributedStep2Master extends AnalysisDistributed {
    public DistributedStep2MasterInput input;       /*!< %Input data */
    public TrainingMethod              method;      /*!< Computation method for the algorithm */
    private Precision                  prec;        /*!< Precision of intermediate computations */
    public TrainingParameter           parameter;   /*!< Parameters of the algorithm */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the neural network training algorithm by copying input objects and parameters
     * of another neural network training algorithm
     * @param context   Context to manage algorithm
     * @param other     An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    public DistributedStep2Master(DaalContext context, DistributedStep2Master other) {
        super(context);
        method = other.method;
        prec = other.prec;
        cObject = cClone(other.cObject, prec.getValue(), method.getValue());
        input = new DistributedStep2MasterInput(getContext(), cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new TrainingParameter(getContext(), cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs the neural network training algorithm
     * @param context   Context to manage algorithm
     * @param cls       Data type to use in intermediate computations for the algorithm,
     *                  Double.class or Float.class
     * @param method    Computation method of the algorithm, @ref TrainingMethod
     * @param optimizationSolver Optimization solver
     */
    public DistributedStep2Master(DaalContext context, Class<? extends Number> cls, TrainingMethod method, Batch optimizationSolver) {
        super(context);
        initialize(context, cls, method, optimizationSolver);
    }

    /**
     * Constructs neural network model-based training object with Float data type used for intermediate computations and default computation method
     * @param context   Context to manage the neural network training object
     * @param optimizationSolver Optimization solver
     */
    public DistributedStep2Master(DaalContext context, Batch optimizationSolver) {
        super(context);
        initialize(context, Float.class, TrainingMethod.defaultDense, optimizationSolver);
    }

    private void initialize(DaalContext context, Class<? extends Number> cls, TrainingMethod method, Batch optimizationSolver) {
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

        cObject = cInit(prec.getValue(), method.getValue(), optimizationSolver.cObject);
        input = new DistributedStep2MasterInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new TrainingParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Initializes the neural network topology of the layers
     * @param dataSize    Size of the input data for the trainings
     * @param topology    TrainingTopology of the layers
     */
    public void initialize(long[] dataSize, TrainingTopology topology) {
        cInitialize(cObject, prec.getValue(), method.getValue(), dataSize, topology.cObject);
    }

    /**
     * Runs the neural network training algorithm on master
     * @return  Partial results of the neural network training algorithm
     */
    @Override
    public DistributedPartialResult compute() {
        super.compute();
        return new DistributedPartialResult(getContext(), cGetPartialResult(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Computes the results of the neural network training algorithm on master
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
    public void setPartialResult(DistributedPartialResult partialResult) {
        cSetPartialResult(cObject, prec.getValue(), method.getValue(), partialResult.getCObject());
    }

    /**
     * Registers user-allocated memory to store the results of the neural network training algorithm
     * @return  Results of the neural network training algorithm
     */
    public TrainingResult getResult() {
        return new TrainingResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Returns the newly allocated neural network training algorithm with a copy of input objects
     * and parameters of this neural network training algorithm
     * @param context   Context to manage algorithm
     *
     * @return The newly allocated algorithm
     */
    @Override
    public DistributedStep2Master clone(DaalContext context) {
        return new DistributedStep2Master(context, this);
    }

    private native long cInit(int prec, int method, long optAddr);
    private native long cInitParameter(long addr, int prec, int method);
    private native long cGetInput(long addr, int prec, int method);
    private native long cGetResult(long addr, int prec, int method);
    private native long cGetPartialResult(long addr, int prec, int method);
    private native void cSetPartialResult(long addr, int prec, int method, long cPartialResult);
    private native long cClone(long addr, int prec, int method);
    private native void cInitialize(long algAddr, int prec, int method, long[] dataSize, long configurationAddr);
}
/** @} */
