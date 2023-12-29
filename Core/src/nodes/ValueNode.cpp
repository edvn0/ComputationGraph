#include "nodes/ValueNode.hpp"

namespace Core
{

auto ValueNode::get_weight_statistics() const -> WeightStatistics
{
	const auto is_vector = value.n_cols == 1;
	const double mean = arma::mean(arma::vectorise(value));
	const double stdev = arma::stddev(arma::vectorise(value));
	return {mean, stdev};
}

}  // namespace Core