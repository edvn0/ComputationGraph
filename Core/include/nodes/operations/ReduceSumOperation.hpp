#pragma once

#include "nodes/OperationNode.hpp"
#include <cassert>

namespace Core
{

class ReduceSumOperation : public OperationNode
{
  public:
	~ReduceSumOperation() override = default;
	explicit ReduceSumOperation(const std::vector<Ref<Node>> &inputs,
								std::int32_t chosen = -1)
		: OperationNode(inputs), axis(chosen)
	{
	}
	auto forward(const std::vector<arma::mat> &consumer_outputs)
		-> void override;
	auto backward() -> void override
	{
	}

	auto propagate_gradient(const arma::mat &input) -> std::vector<arma::mat> override;

	auto get_type() const -> NodeType override
	{
		return NodeType::ReduceSum;
	}
	static auto get_operation_type() -> NodeType
	{
		return NodeType::ReduceSum;
	}

	auto get_axis() const -> std::int32_t
	{
		return axis;
	}

  private:
	std::int32_t axis{0};
};

}  // namespace Core