#pragma once

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace Core
{

using u64 = std::uint64_t;
using u32 = std::uint32_t;
using i64 = std::int64_t;
using i32 = std::int32_t;

template <class T> using Ref = std::shared_ptr<T>;
template <class T> using Weak = std::weak_ptr<T>;
template <class T> using Raw = T*;
template <class T> using RefVector = std::vector<Ref<T>>;

}  // namespace Core
