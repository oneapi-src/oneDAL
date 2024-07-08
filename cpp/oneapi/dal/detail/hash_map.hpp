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

#pragma once

#include <functional>
#include "oneapi/dal/detail/common.hpp"

namespace oneapi::dal::detail {

template <typename Key, typename Value>
class hash_map_entry {
public:
    hash_map_entry(const Key& key, const Value& value) : next_(nullptr), key_(key), value_(value) {}

    hash_map_entry* get_next() const {
        return next_;
    }

    const Key& get_key() const {
        return key_;
    }

    const Value& get_value() const {
        return value_;
    }

    void make_next(const Key& key, const Value& value) {
        next_ = new hash_map_entry{ key, value };
    }

    void update(const Key& key, const Value& value) {
        key_ = key;
        value_ = value;
    }

private:
    hash_map_entry* next_;
    Key key_;
    Value value_;
};

template <typename Key, typename Value>
class hash_map {
public:
    using entry_t = hash_map_entry<Key, Value>;
    using entry_ptr = entry_t*;

    explicit hash_map(std::int64_t capacity) : entries_(nullptr), capacity_(capacity) {
        if (capacity <= 0) {
            throw invalid_argument{ error_messages::capacity_leq_zero() };
        }

        entries_ = new entry_ptr[capacity_];
        for (std::int64_t i = 0; i < capacity_; i++) {
            entries_[i] = nullptr;
        }
    }

    ~hash_map() {
        for (std::int64_t i = 0; i < capacity_; i++) {
            entry_ptr current = entries_[i];
            while (current) {
                entry_ptr next = current->get_next();
                delete current;
                current = next;
            }
        }
        delete[] entries_;
        entries_ = nullptr;
        capacity_ = 0;
    }

    hash_map(const hash_map&) = delete;
    hash_map& operator=(const hash_map&) = delete;

    bool has(const Key& key) const {
        return find_entry(key) != nullptr;
    }

    Value get(const Key& key) const {
        const entry_ptr entry = find_entry(key);
        if (!entry) {
            throw invalid_argument{ error_messages::invalid_key() };
        }
        return entry->get_value();
    }

    void set(const Key& key, const Value& value) {
        replace_or_insert(key, value);
    }

private:
    std::int64_t get_index(const Key& key) const {
        return std::hash<Key>{}(key) % capacity_;
    }

    entry_ptr find_entry(const Key& key) const {
        entry_ptr current = entries_[get_index(key)];
        while (current) {
            if (current->get_key() == key) {
                return current;
            }
            current = current->get_next();
        }
        return current;
    }

    void replace_or_insert(const Key& key, const Value& value) {
        const std::int64_t index = get_index(key);

        entry_ptr current = entries_[index];
        if (!current) {
            entries_[index] = new entry_t{ key, value };
            return;
        }

        for (;;) {
            if (current->get_key() == key) {
                return current->update(key, value);
            }
            if (!current->get_next()) {
                return current->make_next(key, value);
            }
            current = current->get_next();
        }
    }

    entry_ptr* entries_;
    std::int64_t capacity_;
};

} // namespace oneapi::dal::detail
