#pragma once

#include "nodes/OperationNode.hpp"
#include <cassert>

namespace Core
{

class NegateOperation : public OperationNode
{
  public:
	~NegateOperation() override = default;
	using OperationNode::OperationNode;

	auto forward(const std::vector<arma::mat> &consumer_outputs)
		-> void override
	{
		assert(consumer_outputs.size() == 1);
		value = -consumer_outputs[0];
	}

	auto backward() -> void override
	{
	}

	auto get_type() const -> NodeType override
	{
		return NodeType::Negate;
	}
	static auto get_operation_type() -> NodeType
	{
		return NodeType::Negate;
	}
};

}  // namespace Core