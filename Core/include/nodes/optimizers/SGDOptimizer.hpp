#pragma once

#include "nodes/OperationNode.hpp"
#include "nodes/optimizers/GradientComputation.hpp"
#include <vector>

namespace Core
{

class SGDOptimizer : public OperationNode
{
  public:
	~SGDOptimizer() override = default;
	explicit SGDOptimizer(const std::vector<Ref<Node>> &inputs,
						  double lr = 0.0001)
		: OperationNode(inputs), learning_rate(lr)
	{
	}

	auto forward() -> void override;
	auto forward(const std::vector<arma::mat> &) -> void override
	{
		forward();
	};

	auto backward() -> void override
	{
	}

	auto get_type() const -> NodeType override
	{
		return NodeType::SGD;
	}

	static auto get_operation_type() -> NodeType
	{
		return NodeType::SGD;
	}

  private:
	double learning_rate{0.001};
};

}  // namespace Core
