/* file: TrainingBatch.java */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/**
 * @brief Contains classes for training the model of the neural network
 */
package com.intel.daal.algorithms.neural_networks.training;

import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.services.DaalContext;

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
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs neural network model-based training object
     * @param context   Context to manage the neural network training object
     * @param cls       Data type to use in intermediate computations for the neural network,
     *                  Double.class or Float.class
     * @param method    Neural network computation method, @ref TrainingMethod
     */
    public TrainingBatch(DaalContext context, Class<? extends Number> cls, TrainingMethod method) {
        super(context);

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
        input = new TrainingInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new TrainingParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
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

    private native long cInit(int prec, int method);
    private native long cInitParameter(long algAddr, int prec, int method);
    private native long cGetInput(long algAddr, int prec, int method);
    private native long cGetResult(long algAddr, int prec, int method);
    private native void cSetResult(long algAddr, int prec, int method, long resAddr);
    private native void cInitialize(long algAddr, int prec, int method, long[] dataSize, long configurationAddr);
    private native long cClone(long algAddr, int prec, int method);
}
