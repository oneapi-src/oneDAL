/* file: Online.java */
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

package com.intel.daal.algorithms.pca;

import com.intel.daal.algorithms.AnalysisOnline;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.PartialResult;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__ONLINE"></a>
 * @brief Runs the PCA algorithm in the online processing mode
 * \n<a href="DAAL-REF-PCA-ALGORITHM">PCA algorithm description and usage models</a>
 *
 * @par References
 *      - ComputeStep class
 *      - Method class
 *      - InputId class
 *      - PartialCorrelationResultID class
 *      - PartialSVDTableResultID class
 *      - PartialSVDCollectionResultID class
 *      - ResultId class
 *      - Input class
 *      - PartialCorrelationResult class
 *      - PartialSVDResult class
 *      - Result class
 */
public class Online extends AnalysisOnline {
    public Input           input;        /*!< %Input data */
    public OnlineParameter parameter;    /*!< Parameter of the algorithm */
    public Method   method;  /*!< Computation method for the algorithm */
    private PartialResult  partialResult; /*!< Partial result of the algorithm */
    private Precision                 prec; /*!< Precision of intermediate computations */

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs the PCA algorithm in the online processing mode
     * by copying input objects and parameters of another PCA algorithm
     * @param context   Context to manage the PCA algorithm
     * @param other     An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    public Online(DaalContext context, Online other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), this.method.getValue());

        parameter = new OnlineParameter(getContext(), cInitParameter(this.cObject, prec.getValue(), method.getValue()),
            cObject, prec, method);
        input = new Input(getContext(), cGetInput(cObject, prec.getValue(), Method.correlationDense.getValue()), ComputeMode.online);
    }

    /**
     * Constructs the PCA algorithm in the online processing mode
     * @param context   Context to manage the PCA algorithm
     * @param cls       Data type to use in intermediate computations for the PCA algorithm,
     *                  Double.class or Float.class
     * @param method    Computation method, @ref Method
     */
    public Online(DaalContext context, Class<? extends Number> cls, Method method) {
        super(context);
        this.method = method;

        if (method != Method.correlationDense && method != Method.svdDense) {
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

        this.cObject = cInit(prec.getValue(), this.method.getValue());

        parameter = new OnlineParameter(getContext(), cInitParameter(this.cObject, prec.getValue(), method.getValue()),
            cObject, prec, method);
        input = new Input(getContext(), cGetInput(cObject, prec.getValue(), Method.correlationDense.getValue()), ComputeMode.online);
    }

    /**
     * Runs the PCA algorithm in the online processing mode
     * @return  Partial results of the PCA algorithm in the online processing mode
     */
    @Override
    public PartialResult compute() {
        super.compute();
        if (method == Method.correlationDense) {
            partialResult = new PartialCorrelationResult(getContext(), cGetPartialResult(cObject, prec.getValue(), method.getValue()));
        }
        else if (method == Method.svdDense) {
            partialResult = new PartialSVDResult(getContext(), cGetPartialResult(cObject, prec.getValue(), method.getValue()));
        }
        else {
            partialResult = null;
        }
        return partialResult;
    }

    /**
     * Computes the results of the PCA algorithm in the online processing mode
     * @return  Results of the PCA algorithm in the online processing mode
     */
    @Override
    public Result finalizeCompute() {
        super.finalizeCompute();
        Result result = new Result(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
        return result;
    }

    /**
     * Registers user-allocated memory to store partial results of the PCA algorithm
     * in the online processing mode
     * @param partialResult         Structure for storing partial results of the PCA algorithm
     * @param initializationFlag    Flag that specifies whether partial results are initialized
     */
    public void setPartialResult(PartialResult partialResult, boolean initializationFlag) {
        this.partialResult = partialResult;
        cSetPartialResult(cObject, prec.getValue(), method.getValue(), partialResult.getCObject(),
                          initializationFlag);
    }

    /**
     * Registers user-allocated memory to store partial results of the PCA algorithm
     * in the online processing mode
     * @param partialResult         Structure for storing partial results of the PCA algorithm
     */
    public void setPartialResult(PartialResult partialResult) {
        setPartialResult(partialResult, false);
    }

    /**
     * Registers user-allocated memory for storing the results of the PCA algorithm
     * in the online processing mode
     * @param result    Structure for storing the results of the PCA algorithm
     */
    public void setResult(Result result) {
        cSetResult(cObject, prec.getValue(), method.getValue(), result.getCObject());
    }

    /**
     * Returns the newly allocated PCA algorithm in the online processing mode
     * with a copy of input objects and parameters of this PCA algorithm
     * @param context   Context to manage the PCA algorithm
     *
     * @return The newly allocated algorithm
     */
    @Override
    public Online clone(DaalContext context) {
        return new Online(context, this);
    }

    private native long cInit(int prec, int method);

    private native long cInitParameter(long algAddr, int prec, int method);

    private native long cGetInput(long algAddr, int prec, int method);

    private native long cGetResult(long algAddr, int prec, int method);

    private native long cGetPartialResult(long algAddr, int prec, int method);

    private native void cSetResult(long cObject, int prec, int method, long cResult);

    private native void cSetPartialResult(long cObject, int prec, int method, long cPartialResult,
                                          boolean initializationFlag);

    private native long cClone(long algAddr, int prec, int method);
}
