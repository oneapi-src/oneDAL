/* file: PredictionBatch.java */
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
 * @brief Contains classes for making prediction based on the trained model
 */
package com.intel.daal.algorithms.neural_networks.prediction;

import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__PREDICTION__PREDICTIONBATCH"></a>
 * \brief Provides methods for neural network model-based prediction in the batch processing mode
 */
public class PredictionBatch extends com.intel.daal.algorithms.Prediction {
    public    PredictionMethod    method;    /*!< Neural network prediction method */
    public    PredictionInput     input;     /*!< %Input data structure */
    public    PredictionParameter parameter; /*!< Prediction parameters */
    protected Precision         prec;      /*!< Data type to use in intermediate computations for neural network model-based prediction */

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs neural network
     * @param context   Context to manage the neural network
     * @param cls       Data type to use in intermediate computations for the neural network,
     *                  Double.class or Float.class
     * @param method    Neural network computation method, @ref PredictionMethod
     */
    public PredictionBatch(DaalContext context, Class<? extends Number> cls, PredictionMethod method) {
        super(context);

        this.method = method;

        if (method != PredictionMethod.defaultDense && method != PredictionMethod.feedforwardDense) {
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
        input = new PredictionInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new PredictionParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs neural network by copying input objects and parameters of another neural network
     * @param context    Context to manage the neural network
     * @param other      A neural network to be used as the source to initialize the input objects
     *                   and parameters of the neural network
     */
    public PredictionBatch(DaalContext context, PredictionBatch other) {
        super(context);
        method = other.method;
        prec = other.prec;
        cObject = cClone(other.cObject, prec.getValue(), method.getValue());
        input = new PredictionInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new PredictionParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Runs the neural network in the batch processing mode
     * @return  Results of the neural network in the batch processing mode
     */
    @Override
    public PredictionResult compute() {
        super.compute();
        PredictionResult result = new PredictionResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
        return result;
    }

    /**
     * Registers user-allocated memory to store the results of the neural network in the batch processing mode
     * @param result Structure for storing the results of the neural network
     */
    public void setResult(PredictionResult result) {
        cSetResult(cObject, prec.getValue(), method.getValue(), result.getCObject());
    }

    /**
     * Returns the newly allocated neural network with a copy of input objects and parameters of this neural network
     * @param context   Context to manage the neural network
     *
     * @return The newly allocated neural network
     */
    @Override
    public PredictionBatch clone(DaalContext context) {
        return new PredictionBatch(context, this);
    }

    private native long cInit(int prec, int method);
    private native long cInitParameter(long algAddr, int prec, int method);
    private native long cGetInput(long algAddr, int prec, int method);
    private native long cGetResult(long algAddr, int prec, int method);
    private native void cSetResult(long algAddr, int prec, int method, long resAddr);
    private native long cClone(long algAddr, int prec, int method);
}
