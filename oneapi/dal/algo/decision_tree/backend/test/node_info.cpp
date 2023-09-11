/*******************************************************************************
* Copyright 2020 Intel Corporation
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
#include "oneapi/dal/algo/decision_tree/common.hpp"
#include "oneapi/dal/algo/decision_tree/backend/node_info_impl.hpp"

namespace oneapi::dal::decision_tree::test {

namespace de = dal::detail;

template <typename Task>
class node_info_test {
    using task_t = Task;
    using node_impl_t = detail::node_info_impl<task_t>;
    using leaf_impl_t = detail::leaf_node_info_impl<task_t>;
    using split_impl_t = detail::split_node_info_impl<task_t>;

public:
    void fill_node(node_info<task_t>& node, bool zeroize = false) {
        auto& impl = dal::detail::cast_impl<node_impl_t>(node);
        impl.level = zeroize ? 0 : level_;
        impl.impurity = zeroize ? 0 : impurity_;
        impl.sample_count = zeroize ? 0 : sample_count_;
    }

    void fill_node(leaf_node_info<task_t>& node, bool zeroize = false) {
        fill_node(static_cast<node_info<task_t>&>(node), zeroize);

        auto& impl = dal::detail::cast_impl<leaf_impl_t>(node);

        if constexpr (std::is_same_v<task_t, task::classification>) {
            impl.response = zeroize ? 0 : response_cls_;

            auto prob_arr = dal::array<double>::empty(impl.class_count);
            vec_prob_.push_back(prob_arr); // node_info doesn't own prob pointer
            auto prob = prob_arr.get_mutable_data();
            for (std::int64_t idx = 0; idx < impl.class_count; idx++) {
                prob[idx] = zeroize ? 0 : static_cast<double>(idx);
            }

            impl.prob = prob; // node_info doesn't own prob pointer
        }
        else {
            impl.response = zeroize ? 0 : response_reg_;
        }
    }

    void fill_node(split_node_info<task_t>& node, bool zeroize = false) {
        fill_node(static_cast<node_info<task_t>&>(node), zeroize);

        auto& impl = dal::detail::cast_impl<split_impl_t>(node);

        impl.feature_index = zeroize ? 0 : feature_index_;
        impl.feature_value = zeroize ? 0 : feature_value_;
    }

    void check_node(const node_info<task_t>& node) {
        REQUIRE(node.get_level() == level_);
        REQUIRE(node.get_impurity() == impurity_);
        REQUIRE(node.get_sample_count() == sample_count_);
    }

    void check_node(const split_node_info<task_t>& node) {
        check_node(static_cast<const node_info<task_t>&>(node));
        REQUIRE(node.get_feature_index() == feature_index_);
        REQUIRE(node.get_feature_value() == feature_value_);
    }

    void check_node(const leaf_node_info<task_t>& node) {
        check_node(static_cast<const node_info<task_t>&>(node));
        if constexpr (std::is_same_v<task_t, task::classification>) {
            REQUIRE(node.get_response() == response_cls_);
            for (std::int64_t idx = 0; idx < class_count_; idx++) {
                REQUIRE(node.get_probability(idx) == static_cast<double>(idx));
            }
        }
        else {
            REQUIRE(node.get_response() == response_reg_);
        }
    }

    template <typename Node>
    void check_assignment_impl(bool move = false) {
        if constexpr (std::is_same_v<Node, leaf_node_info<task::classification>>) {
            Node node_src(class_count_);
            fill_node(node_src);
            INFO("check src fill");
            check_node(node_src);
            Node node_dst(class_count_);
            node_dst = move ? std::move(node_src) : node_src;
            if (!move)
                fill_node(node_src, zero);
            check_node(node_dst);
        }
        else {
            Node node_src;
            fill_node(node_src);
            INFO("check src fill");
            check_node(node_src);
            Node node_dst;
            node_dst = move ? std::move(node_src) : node_src;
            if (!move)
                fill_node(node_src, zero);
            check_node(node_dst);
        }
    }

    void check_assignment(bool move = false) {
        INFO("check assignment");
        check_assignment_impl<node_info<task_t>>(move);
        check_assignment_impl<leaf_node_info<task_t>>(move);
        check_assignment_impl<split_node_info<task_t>>(move);
    }

    template <typename Node>
    void check_construction_impl(bool move = false) {
        if constexpr (std::is_same_v<Node, leaf_node_info<task::classification>>) {
            Node node_src(class_count_);
            fill_node(node_src);
            Node node_dst(move ? std::move(node_src) : node_src);
            if (!move)
                fill_node(node_src, zero);
            check_node(node_dst);
        }
        else {
            Node node_src;
            fill_node(node_src);
            Node node_dst(move ? std::move(node_src) : node_src);
            if (!move)
                fill_node(node_src, zero);
            check_node(node_dst);
        }
    }

    void check_construction(bool move = false) {
        check_construction_impl<node_info<task_t>>(move);
        check_construction_impl<leaf_node_info<task_t>>(move);
        check_construction_impl<split_node_info<task_t>>(move);
    }

private:
    bool zero = true;
    std::int64_t class_count_ = 7;

    std::int64_t level_ = 5;
    double impurity_ = 0.1234;
    std::int64_t sample_count_ = 555;

    std::int64_t feature_index_ = 77;
    double feature_value_ = 9.12;

    std::int64_t response_cls_ = 4;
    double response_reg_ = 0.5;
    std::vector<dal::array<double>> vec_prob_;
};

constexpr bool move = true;

TEMPLATE_TEST_M(node_info_test,
                "<leaf_, split_>node_info copy construction check",
                "[node_info]",
                task::classification,
                task::regression) {
    this->check_construction();
}

TEMPLATE_TEST_M(node_info_test,
                "<leaf_, split_>node_info move construction check",
                "[node_info]",
                task::classification,
                task::regression) {
    this->check_construction(move);
}

TEMPLATE_TEST_M(node_info_test,
                "<leaf_, split_>node_info copy assignment check",
                "[node_info]",
                task::classification,
                task::regression) {
    this->check_assignment();
}

TEMPLATE_TEST_M(node_info_test,
                "<[leaf_, split_>node_info copy assignment check",
                "[node_info]",
                task::classification,
                task::regression) {
    this->check_assignment(move);
}

TEST("leaf_node_info<task::classification> can be construct with class num param > 1 only",
     "[node_info]") {
    REQUIRE_THROWS_AS(leaf_node_info<task::classification>(1), domain_error);
}

} // namespace oneapi::dal::decision_tree::test
