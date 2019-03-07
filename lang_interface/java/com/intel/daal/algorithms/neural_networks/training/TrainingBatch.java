/* file: TrainingBatch.java */
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
 * @defgroup neural_networks_training_batch Batch
 * @ingroup neural_networks_training
 * @{
 */
/**
 * @brief Contains classes for training the model of the neural network
 */
package com.intel.daal.algorithms.neural_networks.training;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.services.DaalContext;
import com.intel.daal.algorithms.optimization_solver.iterative_solver.Batch;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__TRAINING__TRAININGBATCH"></a>
 * \brief Provides methods for neural network model-based training in the batch processing mode
 */
public class TrainingBatch extends com.intel.daal.algorithms.TrainingBatch {
    public    TrainingMethod    method;    /*!< Neural network training method */
    public    TrainingInput     input;     /*!< %Input data structure */
    public    TrainingParameter parameter; /*!< Training parameters */
    protected Precision         prec;      /*!< Data type to use in intermediate computations for neural network model-based training */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs neural network model-based training object
     * @param context   Context to manage the neural network training object
     * @param cls       Data type to use in intermediate computations for the neural network,
     *                  Double.class or Float.class
     * @param method    Neural network computation method, @ref TrainingMethod
     * @param optimizationSolver Optimization solver
     */
    public TrainingBatch(DaalContext context, Class<? extends Number> cls, TrainingMethod method, Batch optimizationSolver) {
        super(context);
        initialize(context, cls, method, optimizationSolver);
    }

    /**
     * Constructs neural network model-based training object with default computation method
     * @param context   Context to manage the neural network training object
     * @param cls       Data type to use in intermediate computations for the neural network,
     *                  Double.class or Float.class
     * @param optimizationSolver Optimization solver
     */
    public TrainingBatch(DaalContext context, Class<? extends Number> cls, Batch optimizationSolver) {
        super(context);
        initialize(context, cls, TrainingMethod.defaultDense, optimizationSolver);
    }

    /**
     * Constructs neural network model-based training object with Float data type used for intermediate computations and default computation method
     * @param context   Context to manage the neural network training object
     * @param optimizationSolver Optimization solver
     */
    public TrainingBatch(DaalContext context, Batch optimizationSolver) {
        super(context);
        initialize(context, Float.class, TrainingMethod.defaultDense, optimizationSolver);
    }

    /**
     * Constructs neural network by copying input objects and parameters of another neural network
     * @param context    Context to manage the neural network
     * @param other      A neural network to be used as the source to initialize the input objects
     *                   and parameters of the neural network
     */
    public TrainingBatch(DaalContext context, TrainingBatch other) {
        super(context);
        method = other.method;
        prec = other.prec;
        cObject = cClone(other.cObject, prec.getValue(), method.getValue());
        input = new TrainingInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
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
     * Runs the neural network in the batch processing mode
     * @return  Results of the neural network in the batch processing mode
     */
    @Override
    public TrainingResult compute() {
        super.compute();
        TrainingResult result = new TrainingResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
        return result;
    }

    /**
     * Registers user-allocated memory to store the results of the neural network in the batch processing mode
     * @param result Structure for storing the results of the neural network
     */
    public void setResult(TrainingResult result) {
        cSetResult(cObject, prec.getValue(), method.getValue(), result.getCObject());
    }

    /**
     * Returns the newly allocated neural network with a copy of input objects and parameters of this neural network
     * @param context   Context to manage the neural network
     *
     * @return The newly allocated neural network
     */
    @Override
    public TrainingBatch clone(DaalContext context) {
        return new TrainingBatch(context, this);
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
        input = new TrainingInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new TrainingParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    private native long cInit(int prec, int method, long optSolverAddr);
    private native long cInitParameter(long algAddr, int prec, int method);
    private native long cGetInput(long algAddr, int prec, int method);
    private native long cGetResult(long algAddr, int prec, int method);
    private native void cSetResult(long algAddr, int prec, int method, long resAddr);
    private native void cInitialize(long algAddr, int prec, int method, long[] dataSize, long configurationAddr);
    private native long cClone(long algAddr, int prec, int method);
}
/** @} */
