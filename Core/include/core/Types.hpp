#pragma once

#include <memory>
#include <vector>
#include <utility>

namespace Core {

template<class T> using Ref = std::shared_ptr<T>;
template<class T> using Weak = std::weak_ptr<T>;
template<class T> using RefVector = std::vector<Ref<T>>;

}
