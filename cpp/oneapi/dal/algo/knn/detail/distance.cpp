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

#include "oneapi/dal/algo/knn/detail/distance.hpp"
#include "oneapi/dal/algo/knn/backend/distance_impl.hpp"

namespace oneapi::dal::knn::detail {

template <typename F, typename M>
using minkowski_distance_t = minkowski_distance::descriptor<F, M>;

template <typename F, typename M>
using chebyshev_distance_t = chebyshev_distance::descriptor<F, M>;

template <typename F, typename M>
using cosine_distance_t = cosine_distance::descriptor<F, M>;

class daal_interop_chebyshev_distance_impl : public distance_impl {
public:
    daal_interop_chebyshev_distance_impl() = default;

    daal_distance_t get_daal_distance_type() override {
        return daal_distance_t::chebyshev;
    }

    double get_degree() override {
        return 0.0;
    }
};

class daal_interop_minkowski_distance_impl : public distance_impl {
public:
    daal_interop_minkowski_distance_impl(double degree) : degree_(degree) {}

    daal_distance_t get_daal_distance_type() override {
        return daal_distance_t::minkowski;
    }

    double get_degree() override {
        return degree_;
    }

private:
    double degree_;
};

class daal_interop_cosine_distance_impl : public distance_impl {
public:
    daal_interop_cosine_distance_impl() = default;

    daal_distance_t get_daal_distance_type() override {
        return daal_distance_t::cosine;
    }

    double get_degree() override {
        return 0.0;
    }
};

template <typename F, typename M>
distance<minkowski_distance_t<F, M>>::distance(const minkowski_distance_t<F, M> &dist)
        : distance_(dist),
          impl_(new daal_interop_minkowski_distance_impl{ dist.get_degree() }) {}

template <typename F, typename M>
distance_impl *distance<minkowski_distance_t<F, M>>::get_impl() const {
    return impl_.get();
}

template <typename F, typename M>
distance<chebyshev_distance_t<F, M>>::distance(const chebyshev_distance_t<F, M> &dist)
        : distance_(dist),
          impl_(new daal_interop_chebyshev_distance_impl{}) {}

template <typename F, typename M>
distance_impl *distance<chebyshev_distance_t<F, M>>::get_impl() const {
    return impl_.get();
}

template <typename F, typename M>
distance<cosine_distance_t<F, M>>::distance(const cosine_distance_t<F, M> &dist)
        : distance_(dist),
          impl_(new daal_interop_cosine_distance_impl{}) {}

template <typename F, typename M>
distance_impl *distance<cosine_distance_t<F, M>>::get_impl() const {
    return impl_.get();
}

#define INSTANTIATE_MINKOWSKI(F, M) \
    template class ONEDAL_EXPORT distance<minkowski_distance_t<F, M>>;

#define INSTANTIATE_CHEBYSHEV(F, M) \
    template class ONEDAL_EXPORT distance<chebyshev_distance_t<F, M>>;

#define INSTANTIATE_COSINE(F, M) template class ONEDAL_EXPORT distance<cosine_distance_t<F, M>>;

INSTANTIATE_MINKOWSKI(float, minkowski_distance::method::dense)
INSTANTIATE_MINKOWSKI(double, minkowski_distance::method::dense)

INSTANTIATE_CHEBYSHEV(float, chebyshev_distance::method::dense)
INSTANTIATE_CHEBYSHEV(double, chebyshev_distance::method::dense)

INSTANTIATE_COSINE(float, cosine_distance::method::dense)
INSTANTIATE_COSINE(double, cosine_distance::method::dense)

} // namespace oneapi::dal::knn::detail
