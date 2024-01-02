#include "nodes/optimizers/GradientComputation.hpp"
#include <queue>
#include <unordered_map>
#include <unordered_set>

#include "nodes/Node.hpp"

namespace Core
{

auto compute_gradients(Ref<Node> &loss) -> std::unordered_map<Node *, arma::mat>
{
	std::unordered_map<Node *, arma::mat> grad_table;
	grad_table[loss.get()] =
		arma::mat(1, 1, arma::fill::ones);	// Assuming loss gradient is 1

	std::unordered_set<std::shared_ptr<Node>> visited;
	std::queue<std::shared_ptr<Node>> queue;
	visited.insert(loss);
	queue.push(loss);

	while (!queue.empty())
	{
		auto node = queue.front();
		queue.pop();

		if (node != loss)
		{
			auto &current_gradient = grad_table[node.get()];
			current_gradient.set_size(1, 1);
			current_gradient.at(0) = 0;

			for (auto &weak_consumer : node->consumers)
			{
				auto consumer = weak_consumer.lock();
				if (!consumer)
					continue;

				const auto &lossgrad_wrt_consumer_output =
					grad_table.at(consumer.get());

				// Assuming a backward method for each operation type
				auto lossgrads_wrt_consumer_inputs =
					consumer->propagate_gradient(lossgrad_wrt_consumer_output);

				if (consumer->inputs.size() == 1)
				{
					// We've got a scalar
					current_gradient += lossgrads_wrt_consumer_inputs.at(0);
				}
				else
				{
					const auto it = std::find(consumer->inputs.begin(),
											  consumer->inputs.end(), node);
					const auto node_index_in_consumer_inputs =
						std::distance(consumer->inputs.begin(), it);
					const auto &lossgrad_wrt_node =
						lossgrads_wrt_consumer_inputs.at(
							node_index_in_consumer_inputs);
					current_gradient += lossgrad_wrt_node;
				}
			}
		}

		if (node->inputs.empty())
			continue;

		for (auto &input_node : node->inputs)
		{
			if (visited.contains(input_node))
				continue;

			visited.insert(input_node);
			queue.push(input_node);
		}
	}

	return grad_table;
}

}  // namespace Core
