#include "nodes/optimizers/SGDOptimizer.hpp"

#include "nodes/optimizers/GradientComputation.hpp"

namespace Core
{
auto SGDOptimizer::forward() -> void
{
	auto computed_gradient = compute_gradients(get_root());

	for (auto &&[k, v] : computed_gradient)
	{
		if (k->get_type() == NodeType::Value)
		{
			const arma::mat computed = v * learning_rate;
			k->value -= computed;
		}
	}
}

}  // namespace Core
