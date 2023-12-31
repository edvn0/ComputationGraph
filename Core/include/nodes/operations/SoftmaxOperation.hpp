#pragma once

#include "nodes/OperationNode.hpp"
#include <cassert>

namespace Core
{

class SoftmaxOperation : public OperationNode
{
  public:
	using OperationNode::OperationNode;
	~SoftmaxOperation() override = default;

	auto forward(const std::vector<arma::mat> &consumer_outputs)
		-> void override;
	auto backward() -> void override
	{
	}

	auto get_type() const -> NodeType override
	{
		return NodeType::Softmax;
	}
	static auto get_operation_type() -> NodeType
	{
		return NodeType::Softmax;
	}
};

}  // namespace Core