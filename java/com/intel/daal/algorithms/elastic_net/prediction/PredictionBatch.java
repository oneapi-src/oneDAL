/* file: PredictionBatch.java */
/*******************************************************************************
* Copyright 2014-2022 Intel Corporation
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
 * @defgroup elastic_net_prediction_batch Batch
 * @ingroup elastic_net_prediction
 * @{
 */
package com.intel.daal.algorithms.elastic_net.prediction;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__ELASTIC_NET__PREDICTION__PREDICTIONBATCH"></a>
 * @brief Provides methods for elastic net model-based prediction
 * <!-- \n<a href="DAAL-REF-ELASTICNET-ALGORITHM">Elastic net algorithm description and usage models</a> -->
 *
 * @par References
 *      - Model class
 *      - ModelNormEq class
 *      - PredictionInputId class
 *      - PredictionResultId class
 */
public class PredictionBatch extends com.intel.daal.algorithms.Prediction {
    public Input            input;  /*!< %Input data */
    public PredictionMethod method; /*!< %Prediction method for the algorithm */

    private Precision       prec;   /*!< Precision of intermediate computations */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs a elastic net prediction algorithm by copying
     * input objects and parameters of another elastic net prediction algorithm
     * @param context   Context to manage elastic net model-based prediction
     * @param other     %Algorithm to use as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    public PredictionBatch(DaalContext context, PredictionBatch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), method.getValue());
        input = new Input(getContext(), cGetInput(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs the elastic net prediction algorithm in the batch processing mode
     * @param context   Context to manage elastic net model-based prediction
     * @param cls       Data type to use in intermediate computations of elastic net,
     *                  Double.class or Float.class
     * @param method    %Algorithm prediction method, @ref PredictionMethod
     */
    public PredictionBatch(DaalContext context, Class<? extends Number> cls, PredictionMethod method) {
        super(context);

        this.method = method;
        if (this.method != PredictionMethod.defaultDense) {
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
    }

    /**
     * Computes the result of elastic net model-based prediction in the batch processing mode
     * @return Result of elastic net model-based prediction
     */
    @Override
    public PredictionResult compute() {
        super.compute();
        return new PredictionResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Registers user-allocated memory to store the result of elastic net model-based prediction
     * @param result Object to store the result of elastic net model-based prediction
     */
    public void setResult(PredictionResult result) {
        cSetResult(cObject, prec.getValue(), method.getValue(), result.getCObject());
    }

    /**
     * Returns a newly allocated elastic net prediction algorithm
     * with a copy of the input objects of this elastic net prediction algorithm
     * in the batch processing mode
     * @param context   Context to manage elastic net model-based prediction
     *
     * @return Newly allocated algorithm
     */
    @Override
    public PredictionBatch clone(DaalContext context) {
        return new PredictionBatch(context, this);
    }

    private native long cInit(int prec, int method);

    private native long cGetInput(long algAddr, int prec, int method);

    private native void cSetResult(long cAlgorithm, int prec, int method, long cObject);

    private native long cGetResult(long algAddr, int prec, int method);

    private native long cClone(long algAddr, int prec, int method);
}
/** @} */
