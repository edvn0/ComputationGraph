#pragma once

#include "nodes/OperationNode.hpp"
#include "nodes/optimizers/GradientComputation.hpp"

namespace Core
{

class SGDOptimizer : public OptimizerNode
{
  public:
	~SGDOptimizer() override = default;
	explicit SGDOptimizer(const Ref<Node> &inputs, double lr = 0.0001)
		: OptimizerNode(inputs), learning_rate(lr)
	{
	}

	auto forward() -> void override;

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
