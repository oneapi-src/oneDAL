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

#pragma once

template <typename Container>
class enumerate_iterator {
public:
    using container_const_iterator_t = typename Container::const_iterator;
    using value_t = typename Container::value_type;

    explicit enumerate_iterator(std::size_t position, const container_const_iterator_t& iterator)
            : position_(position),
              iterator_(iterator) {}

    bool is_equal(const enumerate_iterator& other) const {
        return iterator_ == other.iterator_;
    }

    enumerate_iterator& operator++() {
        ++position_;
        ++iterator_;
        return *this;
    }

    enumerate_iterator operator++(int) {
        const auto copy = *this;
        ++position_;
        ++iterator_;
        return copy;
    }

    std::tuple<std::size_t, value_t> operator*() const {
        return std::make_tuple(position_, *iterator_);
    }

    std::tuple<std::size_t, value_t> operator->() const {
        return std::make_tuple(position_, *iterator_);
    }

private:
    std::size_t position_;
    container_const_iterator_t iterator_;
};

template <typename Container>
inline bool operator==(const enumerate_iterator<Container>& lhs,
                       const enumerate_iterator<Container>& rhs) {
    return lhs.is_equal(rhs);
}

template <typename Container>
inline bool operator!=(const enumerate_iterator<Container>& lhs,
                       const enumerate_iterator<Container>& rhs) {
    return !lhs.is_equal(rhs);
}

template <typename Container>
class enumerate {
public:
    explicit enumerate(const Container& container) : container_(container) {}
    explicit enumerate(Container&& container) : container_(std::move(container)) {}

    enumerate_iterator<Container> begin() const {
        return enumerate_iterator<Container>{ 0, container_.begin() };
    }

    enumerate_iterator<Container> end() const {
        return enumerate_iterator<Container>{ container_.size(), container_.end() };
    }

private:
    Container container_;
};
