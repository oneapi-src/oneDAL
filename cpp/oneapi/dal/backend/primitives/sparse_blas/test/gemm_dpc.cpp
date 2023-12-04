/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include "oneapi/dal/backend/primitives/sparse_blas.hpp"
#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"

namespace oneapi::dal::backend::primitives::test {

namespace te = dal::test::engine;

template <ndorder order>
struct order_tag {
    static constexpr ndorder value = order;
};

using c_order = order_tag<ndorder::c>;
using f_order = order_tag<ndorder::f>;

template <transpose t>
struct transpose_tag {
    static constexpr transpose value = t;
};

using transpose_nontrans = transpose_tag<transpose::nontrans>;
using transpose_trans = transpose_tag<transpose::trans>;

template <typename Param>
class sparse_gemm_test : public te::float_algo_fixture<std::tuple_element_t<0, Param>> {
public:
    using float_t = std::tuple_element_t<0, Param>;
    static constexpr transpose trans_a = std::tuple_element_t<1, Param>::value;
    static constexpr ndorder bo = std::tuple_element_t<2, Param>::value;
    static constexpr ndorder co = std::tuple_element_t<3, Param>::value;

    sparse_gemm_test() {
        m_ = 0;
        k_ = 0;
        p_ = 0;
    }

    void generate_dimensions() {
        m_ = GENERATE(4);
        k_ = GENERATE(4);
        p_ = GENERATE(2); // GENERATE(2, 4, 8);
        CAPTURE(m_, k_, p_);
    }

    auto B() {
        check_if_initialized();
        return ndarray<float_t, 2, bo>::ones(this->get_queue(), { k_, p_ });
    }

    auto Bt() {
        check_if_initialized();
        return ndarray<float_t, 2, bo>::ones(this->get_queue(), { p_, k_ });
    }

    auto C() {
        check_if_initialized();
        return ndarray<float_t, 2, co>::empty(this->get_queue(), { m_, p_ });
    }

    void test_gemm(sparse_matrix_handle& a) {
        auto c = C();

        if (bo == ndorder::c) {
            auto [b, b_e] = B();
            gemm(this->get_queue(), trans_a, a, b, c, { b_e }).wait_and_throw();
        }
        /*        else {
            auto [bt, bt_e] = Bt();
            gemm(this->get_queue(), trans_a, a, bt.t(), c, { bt_e }).wait_and_throw();
        }
*/
        check_ones_matrix(c);
    }

    void check_ones_matrix(const ndarray<float_t, 2, co>& mat) {
        check_if_initialized();
        REQUIRE(mat.get_shape() == ndshape<2>{ k_, p_ });

        const float_t* mat_ptr = mat.get_data();
        for (std::int64_t i = 0; i < mat.get_count(); i++) {
            const std::int64_t result = k_ / 2;
            if (std::int64_t(mat_ptr[i]) != result) {
                CAPTURE(i, mat_ptr[i]);
                for (int k = 0; k < k_; k++) {
                    for (int j = 0; j < p_; j++) {
                        std::cout << mat_ptr[k * p_ + j] << ", ";
                    }
                    std::cout << std::endl;
                }
                std::cout << std::endl;
                FAIL();
            }
        }
        SUCCEED();
    }

    bool is_initialized() const {
        return m_ > 0 && k_ > 0 && p_ > 0;
    }

    void check_if_initialized() {
        if (!is_initialized()) {
            throw std::runtime_error{ "sparse gemm test is not initialized" };
        }
    }

private:
    std::int64_t m_;
    std::int64_t k_;
    std::int64_t p_;
};

using gemm_types = COMBINE_TYPES(
    (float, double),
    (transpose_nontrans, transpose_trans),
    (c_order /*, f_order */), /// oneMKL 2024.0 throws 'unimplemented' exception when the matrix B is transposed
    (c_order /*, f_order */));

/// Tests dal::csr_accessor class on a fixed data table:
///     | 1,  0,  1,  0 |
/// A = | 0,  1,  0,  1 |
///     | 1,  0,  1,  0 |
///     | 0,  1,  0,  1 |
TEMPLATE_LIST_TEST_M(sparse_gemm_test,
                     "ones matrix sparse CSR gemm on small sizes",
                     "[csr][gemm][small]",
                     gemm_types) {
    // DPC++ Sparse GEMM from micro MKL libs is not supported on CPU
    SKIP_IF(this->get_policy().is_cpu());

    // Test takes too long time if HW emulates float64
    SKIP_IF(this->not_float64_friendly());
    auto& q = this->get_queue();
    constexpr std::int64_t row_count{ 4 };
    constexpr std::int64_t column_count{ 4 };
    constexpr std::int64_t element_count{ 8 };

    const float data_host[] = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };
    const std::int64_t column_indices_host[] = { 1, 3, 2, 4, 1, 3, 2, 4 };
    const std::int64_t row_offsets_host[] = { 1, 3, 5, 7, 9 };

    auto* const data = sycl::malloc_device<float>(element_count, q);
    auto* const column_indices = sycl::malloc_device<std::int64_t>(element_count, q);
    auto* const row_offsets = sycl::malloc_device<std::int64_t>(row_count + 1, q);

    auto data_event = q.submit([&](sycl::handler& cgh) {
        cgh.memcpy(data, data_host, element_count * sizeof(float));
    });

    auto column_indices_event = q.submit([&](sycl::handler& cgh) {
        cgh.memcpy(column_indices, column_indices_host, element_count * sizeof(std::int64_t));
    });

    auto row_offsets_event = q.submit([&](sycl::handler& cgh) {
        cgh.memcpy(row_offsets, row_offsets_host, (row_count + 1) * sizeof(std::int64_t));
    });

    sparse_matrix_handle handle;

    set_csr_data(q,
                 handle,
                 row_count,
                 column_count,
                 sparse_indexing::one_based,
                 data,
                 column_indices,
                 row_offsets,
                 { data_event, column_indices_event, row_offsets_event })
        .wait();

    this->generate_dimensions();
    this->test_gemm(handle);

    sycl::free(data, q);
    sycl::free(column_indices, q);
    sycl::free(row_offsets, q);
}

} // namespace oneapi::dal::backend::primitives::test
