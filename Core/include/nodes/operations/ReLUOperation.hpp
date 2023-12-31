#pragma once

#include "nodes/OperationNode.hpp"
#include <cassert>

namespace Core
{

class ReLUOperation : public OperationNode
{
  public:
	~ReLUOperation() override = default;
	using OperationNode::OperationNode;

	auto forward(const std::vector<arma::mat> &consumer_outputs)
		-> void override;
	auto backward() -> void override
	{
	}

	auto get_type() const -> NodeType override
	{
		return NodeType::ReLU;
	}
	static auto get_operation_type() -> NodeType
	{
		return NodeType::ReLU;
	}
};

}  // namespace Core