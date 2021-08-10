#include <array>
#include <cmath>
#include <type_traits>

#include "oneapi/dal/test/engine/common_gbench.hpp"
#include "oneapi/dal/test/engine/fixtures_gbench.hpp"
#include "oneapi/dal/test/engine/dataframe.hpp"

#include "oneapi/dal/table/row_accessor.hpp"

#include "oneapi/dal/backend/primitives/reduction/reduction_rm_cw_dpc.hpp"

namespace oneapi::dal::backend::primitives::test {

namespace te = dal::test::engine;
namespace pr = oneapi::dal::backend::primitives;

constexpr auto rm_order = ndorder::c;

using reduction_types = std::tuple<float, sum<float>, square<float>>;

template <typename Param>
class reduction_rm_gbench_fixture : public te::gbench_fixture <std::tuple_element_t<0, Param>> { 
public:
    using float_t = std::tuple_element_t<0, Param>;
    using binary_t = std::tuple_element_t<1, Param>;
    using unary_t = std::tuple_element_t<2, Param>;

    bool is_initialized() const {
        return width_ > 0 && stride_ > 0 && height_ > 0;
    }

    void check_if_initialized() {
        if (!is_initialized()) {
            throw std::runtime_error{ "reduce test is not initialized" };
        }
    }

    bool should_be_skipped() {
        if (width_ > stride_) {
            return true;
        }
        return false;
    }

    auto input() {
        check_if_initialized();
        return ndarray<float_t, 2, rm_order>::zeros(this->get_queue(),
                                                    { stride_, height_ },
                                                    sycl::usm::alloc::device);
    }

    auto output(std::int64_t size) {
        check_if_initialized();
        return ndarray<float_t, 1, rm_order>::zeros(this->get_queue(),
                                                    { size },
                                                    sycl::usm::alloc::device);
    }

    auto fpt_desc() {
        if constexpr (std::is_same_v<float_t, float>) {
            return "float";
        }
        else if constexpr (std::is_same_v<float_t, double>) {
            return "double";
        }
        else return "unknown type";
    }

    auto type_desc() {
        return fmt::format("Floating Point Type: {}", fpt_desc());
    }

    auto matrix_desc() {
        check_if_initialized();
        return fmt::format("Row-Major Matrix with parameters: "
                           "width_ = {}, stride_ = {}, height_ = {}",
                           width_,
                           stride_,
                           height_);
    }

    auto unary_desc() {
        if (std::is_same_v<identity<float_t>, unary_t>) {
            return "Identity Unary Functor";
        }
        else if (std::is_same_v<abs<float_t>, unary_t>) {
            return "Abs Unary Functor";
        }
        else if (std::is_same_v<square<float_t>, unary_t>) {
            return "Square Unary Functor";
        }
        else return "Unknown Unary Functor";
    }

    auto binary_desc() {
        if (std::is_same_v<sum<float_t>, binary_t>) {
            return "Sum Binary Functor";
        }
        else if (std::is_same_v<max<float_t>, binary_t>) {
            return "Max Binary Functor";
        }
        else if (std::is_same_v<min<float_t>, binary_t>) {
            return "Min Binary Functor";
        }
        else return "Unknown Binary Functor";
    }

    auto desc() { 
        return fmt::format("{}; {}; {}; {}",
                           matrix_desc(),
                           type_desc(),
                           unary_desc(),
                           binary_desc());
    } 

    void generate(std::int64_t width, std::int64_t height, std::int64_t stride) { 
	    this->width_ = width;
        this->stride_ = stride;
        this->height_ = height;
    }
    
    void generate(const ::benchmark::State& st) final {
        this->generate(st.range(0), st.range(1), st.range(2));
    }

    std::int64_t get_width() const { return this->width_; }
    std::int64_t get_height() const { return this->height_; }
    std::int64_t get_stride() const { return this->stride_; }

    void run_benchmark(::benchmark::State& st) final {
        auto [inp_array, inp_event] = input();
        auto [out_array, out_event] = output(this->get_width());

        const float_t* inp_ptr = inp_array.get_data();
        float_t* out_ptr = out_array.get_mutable_data();

        this->get_queue().wait_and_throw();

        for(auto _ : st) { 
            reduction_rm_cw_naive<float_t, binary_t, unary_t> reducer(this->get_queue());
            reducer(inp_ptr, out_ptr, get_width(), get_height(), get_stride(), binary_t{}, unary_t{}).wait_and_throw();
        }
    }

private:
    std::int64_t width_;
    std::int64_t stride_;
    std::int64_t height_;
}; 

DEFINE_TEMPLATE_F(reduction_rm_gbench_fixture, BM_rm_cw_reduction_naive, reduction_types)
BENCHMARK_REGISTER_F(reduction_rm_gbench_fixture, BM_rm_cw_reduction_naive)->ArgsProduct({{28, 256, 512, 2000}, 
                                                                                {28, 256, 512, 2000}, {1024, 8192, 32768}});

} // oneapi::dal::backend::primitives::test