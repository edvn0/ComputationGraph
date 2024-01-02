#pragma once

#include "core/Types.hpp"
#include <armadillo>
#include <unordered_map>

namespace Core
{
class Node;

auto compute_gradients(Ref<Node> &root)
	-> std::unordered_map<Node *, arma::mat>;

}  // namespace Core
