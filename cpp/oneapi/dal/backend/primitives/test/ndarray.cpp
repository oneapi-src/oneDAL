
/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"

namespace oneapi::dal::backend::primitives::test {

#define ENUMERATE_AXIS_COUNT_123 ((std::int64_t axis_count), axis_count), 1, 2, 3

class ndarray_test {
public:
    template <typename T, std::int64_t axis_count>
    void check_if_all_equal(const ndarray<T, axis_count>& x, const T& value) const {
        const T* x_ptr = x.get_data();
        for (std::int64_t i = 0; i < x.get_count(); i++) {
            if (x_ptr[i] != value) {
                FAIL();
            }
        }
    }

    template <typename T, std::int64_t axis_count>
    void check_if_all_zeros(const ndarray<T, axis_count>& x) const {
        check_if_all_equal(x, T(0));
    }

    template <typename T, std::int64_t axis_count>
    void check_if_all_ones(const ndarray<T, axis_count>& x) const {
        check_if_all_equal(x, T(1));
    }
};

TEST("ndarray returns correct shapes", "[ndarray_base]") {
    SECTION("1D") {
        const auto x = ndarray_base<1>{ { 5 } };

        REQUIRE(x.get_shape()[0] == 5);
    }

    SECTION("2D") {
        const auto x = ndarray_base<2>{ { 5, 8 } };

        REQUIRE(x.get_shape()[0] == 5);
        REQUIRE(x.get_shape()[1] == 8);
    }

    SECTION("3D") {
        const auto x = ndarray_base<3>{ { 5, 8, 1 } };

        REQUIRE(x.get_shape()[0] == 5);
        REQUIRE(x.get_shape()[1] == 8);
        REQUIRE(x.get_shape()[2] == 1);
    }
}

TEST("ndarray has correct default strides in c-order", "[ndarray_base]") {
    SECTION("1D") {
        const auto x = ndarray_base<1, ndorder::c>{ { 5 } };

        REQUIRE(x.get_strides()[0] == 1);
    }

    SECTION("2D") {
        const auto x = ndarray_base<2, ndorder::c>{ { 5, 8 } };

        REQUIRE(x.get_strides()[0] == 8);
        REQUIRE(x.get_strides()[1] == 1);
    }

    SECTION("3D") {
        const auto x = ndarray_base<3, ndorder::c>{ { 5, 3, 7 } };

        REQUIRE(x.get_strides()[0] == 21);
        REQUIRE(x.get_strides()[1] == 7);
        REQUIRE(x.get_strides()[2] == 1);
    }
}

TEST("ndarray has correct default strides in f-order", "[ndarray_base]") {
    SECTION("1D") {
        const auto x = ndarray_base<1, ndorder::f>{ { 5 } };

        REQUIRE(x.get_strides()[0] == 1);
    }

    SECTION("2D") {
        const auto x = ndarray_base<2, ndorder::f>{ { 5, 8 } };

        REQUIRE(x.get_strides()[0] == 1);
        REQUIRE(x.get_strides()[1] == 5);
    }

    SECTION("3D") {
        const auto x = ndarray_base<3, ndorder::f>{ { 5, 3, 7 } };

        REQUIRE(x.get_strides()[0] == 1);
        REQUIRE(x.get_strides()[1] == 5);
        REQUIRE(x.get_strides()[2] == 15);
    }
}

TEMPLATE_SIG_TEST("can create empty ndview", "[ndview]", ENUMERATE_AXIS_COUNT_123) {
    const auto x = ndview<float, axis_count>{};

    REQUIRE(x.has_data() == false);
    REQUIRE(x.get_count() == 0);
    REQUIRE(x.get_data() == nullptr);
}

TEMPLATE_SIG_TEST("can wrap data into ndview", "[ndview]", ENUMERATE_AXIS_COUNT_123) {
    float data[] = { 0.1 };
    const auto shape = ndshape<axis_count>::square(1);

    const auto x = ndview<float, axis_count>::wrap(data, shape);

    REQUIRE(x.get_data() == data);
    REQUIRE(x.get_shape() == shape);
}

TEMPLATE_SIG_TEST("can create empty ndarray", "[ndarray]", ENUMERATE_AXIS_COUNT_123) {
    const auto x = ndarray<float, axis_count>{};

    REQUIRE(x.has_data() == false);
    REQUIRE(x.get_count() == 0);
    REQUIRE(x.get_data() == nullptr);
}

TEMPLATE_SIG_TEST("can wrap data into ndarray", "[ndarray]", ENUMERATE_AXIS_COUNT_123) {
    float data[] = { 0.1 };
    const auto shape = ndshape<axis_count>::square(1);
    const auto empty_deleter = dal::detail::empty_delete<float>{};
    const auto data_shared = std::shared_ptr<float>{ data, empty_deleter };
    const auto data_array_mutable = array<float>::wrap(data, shape.get_count());
    const auto data_array_immutable =
        array<float>::wrap(const_cast<const float*>(data), shape.get_count());

    SECTION("raw data") {
        const auto x = ndarray<float, axis_count>::wrap(data, shape);

        REQUIRE(x.get_data() == data);
        REQUIRE(x.get_shape() == shape);
    }

    SECTION("shared pointer") {
        const auto x = ndarray<float, axis_count>::wrap(data_shared, shape);

        REQUIRE(x.get_data() == data);
        REQUIRE(x.get_shape() == shape);
    }

    SECTION("shared pointer rvalue") {
        auto movable_data_shared = data_shared;

        const auto x = ndarray<float, axis_count>::wrap(std::move(movable_data_shared), shape);

        REQUIRE(x.get_data() == data);
        REQUIRE(x.get_shape() == shape);
        REQUIRE(movable_data_shared.get() == nullptr);
    }

    SECTION("immutable array") {
        const auto x = ndarray<const float, axis_count>::wrap(data_array_immutable, shape);

        REQUIRE(x.get_data() == data_array_immutable.get_data());
        REQUIRE(x.get_shape() == shape);

        auto data_array = data_array_immutable;
        data_array.need_mutable_data();
        REQUIRE(x.get_data() != data_array.get_mutable_data());
    }

    SECTION("immutable array rvalue") {
        auto movable_data_array_immutable = data_array_immutable;

        const auto x =
            ndarray<const float, axis_count>::wrap(std::move(movable_data_array_immutable), shape);

        REQUIRE(x.get_data() == data_array_immutable.get_data());
        REQUIRE(x.get_shape() == shape);
        REQUIRE(movable_data_array_immutable.get_data() == nullptr);

        auto data_array = data_array_immutable;
        data_array.need_mutable_data();
        REQUIRE(x.get_data() != data_array.get_mutable_data());
    }

    SECTION("mutable array") {
        const auto x = ndarray<float, axis_count>::wrap_mutable(data_array_mutable, shape);

        REQUIRE(x.get_data() == data_array_mutable.get_data());
        REQUIRE(x.get_data() == data_array_mutable.get_mutable_data());
        REQUIRE(x.get_shape() == shape);
    }

    SECTION("mutable array rvalue") {
        auto movable_data_array_mutable = data_array_mutable;

        const auto x =
            ndarray<float, axis_count>::wrap_mutable(std::move(movable_data_array_mutable), shape);

        REQUIRE(x.get_data() == data_array_mutable.get_data());
        REQUIRE(x.get_data() == data_array_mutable.get_mutable_data());
        REQUIRE(x.get_shape() == shape);
        REQUIRE(movable_data_array_mutable.get_data() == nullptr);
    }
}

TEST("can wrap array into ndarray without shape", "[ndarray]") {
    float data[] = { 0.1 };
    const auto data_array_mutable = array<float>::wrap(data, 1);
    const auto data_array_immutable = array<float>::wrap(const_cast<const float*>(data), 1);

    SECTION("immutable array") {
        const auto x = ndarray<const float, 1>::wrap(data_array_immutable);

        REQUIRE(x.get_data() == data_array_immutable.get_data());
        REQUIRE(x.get_shape() == ndshape<1>{ 1 });

        auto data_array = data_array_immutable;
        data_array.need_mutable_data();
        REQUIRE(x.get_data() != data_array.get_mutable_data());
    }

    SECTION("immutable array rvalue") {
        auto movable_data_array_immutable = data_array_immutable;

        const auto x = ndarray<const float, 1>::wrap(std::move(movable_data_array_immutable));

        REQUIRE(x.get_data() == data_array_immutable.get_data());
        REQUIRE(x.get_shape() == ndshape<1>{ 1 });
        REQUIRE(movable_data_array_immutable.get_data() == nullptr);

        auto data_array = data_array_immutable;
        data_array.need_mutable_data();
        REQUIRE(x.get_data() != data_array.get_mutable_data());
    }

    SECTION("mutable array") {
        const auto x = ndarray<float, 1>::wrap_mutable(data_array_mutable);

        REQUIRE(x.get_data() == data_array_mutable.get_data());
        REQUIRE(x.get_data() == data_array_mutable.get_mutable_data());
        REQUIRE(x.get_shape() == ndshape<1>{ 1 });
    }

    SECTION("mutable array rvalue") {
        auto movable_data_array_mutable = data_array_mutable;

        const auto x = ndarray<float, 1>::wrap_mutable(std::move(movable_data_array_mutable));

        REQUIRE(x.get_data() == data_array_mutable.get_data());
        REQUIRE(x.get_data() == data_array_mutable.get_mutable_data());
        REQUIRE(x.get_shape() == ndshape<1>{ 1 });
        REQUIRE(movable_data_array_mutable.get_data() == nullptr);
    }
}

TEMPLATE_SIG_TEST("can create ndarray with custom deleter", "[ndarray]", ENUMERATE_AXIS_COUNT_123) {
    struct custom_deleter {
        custom_deleter() {
            call_counter = std::make_shared<std::int64_t>(0);
        }

        void operator()(float* ptr) {
            (*call_counter)++;
        }

        std::int64_t get_call_count() const {
            return *call_counter;
        }

        std::shared_ptr<std::int64_t> call_counter;
    };

    float data[] = { 0.1 };
    const auto shape = ndshape<axis_count>::square(1);
    auto deleter = custom_deleter{};

    { const auto x = ndarray<float, axis_count>::wrap(data, shape, deleter); }

    REQUIRE(deleter.get_call_count() == 1);
}

template <template <typename, std::int64_t, ndorder> typename Nd>
void test_nd_transpose() {
    // Allocate enough element count for all shapes under test
    std::vector<float> data_vector(1000, 0.0f);
    float* data_ptr = data_vector.data();

    SECTION("1D") {
        const auto x = Nd<float, 1, ndorder::c>::wrap(data_ptr, { 5 });

        const auto xt = x.t();

        REQUIRE(xt.get_shape()[0] == 5);
        REQUIRE(xt.get_strides()[0] == 1);
    }

    SECTION("2D") {
        const auto x = Nd<float, 2, ndorder::c>::wrap(data_ptr, { 3, 7 });

        const auto xt = x.t();

        REQUIRE(xt.get_shape()[0] == 7);
        REQUIRE(xt.get_shape()[1] == 3);
        REQUIRE(xt.get_strides()[0] == 1);
        REQUIRE(xt.get_strides()[1] == 7);
    }

    SECTION("3D") {
        const auto x = Nd<float, 3, ndorder::c>::wrap(data_ptr, { 5, 3, 7 });

        const auto xt = x.t();

        REQUIRE(xt.get_shape()[0] == 7);
        REQUIRE(xt.get_shape()[1] == 3);
        REQUIRE(xt.get_shape()[2] == 5);
        REQUIRE(xt.get_strides()[0] == 1);
        REQUIRE(xt.get_strides()[1] == 7);
        REQUIRE(xt.get_strides()[2] == 21);
    }
}

template <template <typename, std::int64_t, ndorder> typename Nd>
void test_nd_reshape() {
    // Allocate enough element count for all shapes under test
    std::vector<float> data_vector(1000, 0.0f);
    float* data_ptr = data_vector.data();

    SECTION("1D -> 1D") {
        const auto x = Nd<float, 1, ndorder::c>::wrap(data_ptr, { 5 });

        const auto xr = x.reshape(ndshape<1>{ 5 });

        REQUIRE(xr.get_shape()[0] == 5);
    }

    SECTION("1D -> 2D") {
        const auto x = Nd<float, 1, ndorder::c>::wrap(data_ptr, { 8 });

        const auto xr = x.reshape(ndshape<2>{ 4, 2 });

        REQUIRE(xr.get_shape()[0] == 4);
        REQUIRE(xr.get_shape()[1] == 2);
    }

    SECTION("2D -> 2D") {
        const auto x = Nd<float, 2, ndorder::c>::wrap(data_ptr, { 10, 2 });

        const auto xr = x.reshape(ndshape<2>{ 5, 4 });

        REQUIRE(xr.get_shape()[0] == 5);
        REQUIRE(xr.get_shape()[1] == 4);
    }

    SECTION("3D -> 2D") {
        const auto x = Nd<float, 3, ndorder::c>::wrap(data_ptr, { 8, 4, 2 });

        const auto xr = x.reshape(ndshape<2>{ 8, 8 });

        REQUIRE(xr.get_shape()[0] == 8);
        REQUIRE(xr.get_shape()[1] == 8);
    }
}

TEST("ndview transpose", "[ndview]") {
    test_nd_transpose<ndview>();
}

TEST("ndarray transpose", "[ndarray]") {
    test_nd_transpose<ndarray>();
}

TEST("ndview reshape", "[ndview]") {
    test_nd_reshape<ndview>();
}

TEST("ndarray reshape", "[ndarray]") {
    test_nd_reshape<ndarray>();
}

#ifdef ONEDAL_DATA_PARALLEL

TEST("can allocate empty ndarray", "[ndarray]") {
    DECLARE_TEST_POLICY(policy);
    auto& queue = policy.get_queue();

    const auto x = ndarray<float, 2>::empty(queue, { 7, 5 });

    REQUIRE(x.get_data() != nullptr);
}

TEST_M(ndarray_test, "can allocate zeros ndarray", "[ndarray]") {
    DECLARE_TEST_POLICY(policy);
    auto& queue = policy.get_queue();

    auto [x, event] = ndarray<float, 2>::zeros(queue, { 7, 5 });
    event.wait_and_throw();
    check_if_all_zeros(x);
}

TEST_M(ndarray_test, "can allocate ones ndarray", "[ndarray]") {
    DECLARE_TEST_POLICY(policy);
    auto& queue = policy.get_queue();

    auto [x, event] = ndarray<float, 2>::ones(queue, { 7, 5 });
    event.wait_and_throw();
    check_if_all_ones(x);
}

#endif

} // namespace oneapi::dal::backend::primitives::test
