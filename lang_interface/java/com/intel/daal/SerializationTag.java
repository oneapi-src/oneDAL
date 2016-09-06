/* file: SerializationTag.java */
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
 * @brief Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) package
 */
package com.intel.daal;

/**
 * <a name="DAAL-CLASS-SERIALIZATIONTAG"></a>
 */
public final class SerializationTag {
    private int _value;

    static {
        System.loadLibrary("JavaAPI");
    }

    public SerializationTag(int value) {
        _value = value;
    }

    public int getValue() {
        return _value;
    }

    private static final int SERIALIZATION_HOMOGEN_FLOAT32_NT_ID_VALUE                             = 1000;
    private static final int SERIALIZATION_HOMOGEN_FLOAT64_NT_ID_VALUE                             = 1001;
    private static final int SERIALIZATION_HOMOGEN_INT32_S_NT_ID_VALUE                             = 1002;
    private static final int SERIALIZATION_HOMOGEN_INT32_U_NT_ID_VALUE                             = 1003;
    private static final int SERIALIZATION_HOMOGEN_INT64_S_NT_ID_VALUE                             = 1004;
    private static final int SERIALIZATION_HOMOGEN_INT64_U_NT_ID_VALUE                             = 1005;
    private static final int SERIALIZATION_DATACOLLECTION_ID_VALUE                                 = 4000;
    private static final int SERIALIZATION_KEYVALUEDATACOLLECTION_ID_VALUE                         = 4010;
    private static final int SERIALIZATION_CSR_NT_ID_VALUE                                         = 8000;
    private static final int SERIALIZATION_JAVANIO_NT_ID_VALUE                                     = 10000;
    private static final int SERIALIZATION_IMPLICIT_ALS_PARTIALMODEL_ID_VALUE                      = 101610;
    private static final int SERIALIZATION_IMPLICIT_ALS_PREDICTION_RATINGS_PARTIAL_RESULT_ID_VALUE = 101620;
    private static final int SERIALIZATION_IMPLICIT_ALS_PREDICTION_RATINGS_RESULT_ID_VALUE         = 101630;
    private static final int SERIALIZATION_SVM_MODEL_ID_VALUE                                      = 100800;
    private static final int SERIALIZATION_HOMOGEN_FLOAT32_TENSOR_ID_VALUE                         = 20000;
    private static final int SERIALIZATION_HOMOGEN_FLOAT64_TENSOR_ID_VALUE                         = 20001;
    private static final int SERIALIZATION_HOMOGEN_INT32_S_TENSOR_ID_VALUE                         = 20002;
    private static final int SERIALIZATION_HOMOGEN_INT32_U_TENSOR_ID_VALUE                         = 20003;
    private static final int SERIALIZATION_HOMOGEN_INT64_S_TENSOR_ID_VALUE                         = 20004;
    private static final int SERIALIZATION_HOMOGEN_INT64_U_TENSOR_ID_VALUE                         = 20005;

    public static final SerializationTag SERIALIZATION_HOMOGEN_FLOAT32_NT_ID     = new SerializationTag(SERIALIZATION_HOMOGEN_FLOAT32_NT_ID_VALUE);
    public static final SerializationTag SERIALIZATION_HOMOGEN_FLOAT64_NT_ID     = new SerializationTag(SERIALIZATION_HOMOGEN_FLOAT64_NT_ID_VALUE);
    public static final SerializationTag SERIALIZATION_HOMOGEN_INT32_S_NT_ID     = new SerializationTag(SERIALIZATION_HOMOGEN_INT32_S_NT_ID_VALUE);
    public static final SerializationTag SERIALIZATION_HOMOGEN_INT32_U_NT_ID     = new SerializationTag(SERIALIZATION_HOMOGEN_INT32_U_NT_ID_VALUE);
    public static final SerializationTag SERIALIZATION_HOMOGEN_INT64_S_NT_ID     = new SerializationTag(SERIALIZATION_HOMOGEN_INT64_S_NT_ID_VALUE);
    public static final SerializationTag SERIALIZATION_HOMOGEN_INT64_U_NT_ID     = new SerializationTag(SERIALIZATION_HOMOGEN_INT64_U_NT_ID_VALUE);
    public static final SerializationTag SERIALIZATION_DATACOLLECTION_ID         = new SerializationTag(SERIALIZATION_DATACOLLECTION_ID_VALUE);
    public static final SerializationTag SERIALIZATION_KEYVALUEDATACOLLECTION_ID = new SerializationTag(SERIALIZATION_KEYVALUEDATACOLLECTION_ID_VALUE);
    public static final SerializationTag SERIALIZATION_CSR_NT_ID                 = new SerializationTag(SERIALIZATION_CSR_NT_ID_VALUE);
    public static final SerializationTag SERIALIZATION_JAVANIO_NT_ID             = new SerializationTag(SERIALIZATION_JAVANIO_NT_ID_VALUE);
    public static final SerializationTag SERIALIZATION_SVM_MODEL_ID              = new SerializationTag(SERIALIZATION_SVM_MODEL_ID_VALUE);
    public static final SerializationTag SERIALIZATION_IMPLICIT_ALS_PARTIALMODEL_ID              = new SerializationTag(SERIALIZATION_IMPLICIT_ALS_PARTIALMODEL_ID_VALUE);
    public static final SerializationTag SERIALIZATION_IMPLICIT_ALS_PREDICTION_RATINGS_RESULT_ID = new SerializationTag(SERIALIZATION_IMPLICIT_ALS_PREDICTION_RATINGS_RESULT_ID_VALUE);
    public static final SerializationTag SERIALIZATION_IMPLICIT_ALS_PREDICTION_RATINGS_PARTIAL_RESULT_ID = new SerializationTag(SERIALIZATION_IMPLICIT_ALS_PREDICTION_RATINGS_PARTIAL_RESULT_ID_VALUE);
    public static final SerializationTag SERIALIZATION_HOMOGEN_FLOAT32_TENSOR_ID = new SerializationTag(SERIALIZATION_HOMOGEN_FLOAT32_TENSOR_ID_VALUE);
    public static final SerializationTag SERIALIZATION_HOMOGEN_FLOAT64_TENSOR_ID = new SerializationTag(SERIALIZATION_HOMOGEN_FLOAT64_TENSOR_ID_VALUE);
    public static final SerializationTag SERIALIZATION_HOMOGEN_INT32_S_TENSOR_ID = new SerializationTag(SERIALIZATION_HOMOGEN_INT32_S_TENSOR_ID_VALUE);
    public static final SerializationTag SERIALIZATION_HOMOGEN_INT32_U_TENSOR_ID = new SerializationTag(SERIALIZATION_HOMOGEN_INT32_U_TENSOR_ID_VALUE);
    public static final SerializationTag SERIALIZATION_HOMOGEN_INT64_S_TENSOR_ID = new SerializationTag(SERIALIZATION_HOMOGEN_INT64_S_TENSOR_ID_VALUE);
    public static final SerializationTag SERIALIZATION_HOMOGEN_INT64_U_TENSOR_ID = new SerializationTag(SERIALIZATION_HOMOGEN_INT64_U_TENSOR_ID_VALUE);
}
