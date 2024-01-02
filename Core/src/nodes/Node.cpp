#include "nodes/Node.hpp"
#include "core/Typelist.hpp"
#include "network/OperationTypelist.hpp"

#include <fmt/format.h>

namespace Core
{

template <typename Typelist> struct IsOperationInTypelist;

template <typename First, typename... Rest>
struct IsOperationInTypelist<Typelist<First, Rest...>>
{
	static bool check(NodeType type)
	{
		return type == First::get_operation_type() ||
			   IsOperationInTypelist<Typelist<Rest...>>::check(type);
	}
};

template <> struct IsOperationInTypelist<Typelist<>>
{
	static bool check(NodeType)
	{
		return false;
	}
};

auto to_string(NodeType type) -> std::string
{
	switch (type)
	{
	case NodeType::Activation:
		return "Activation";
	case NodeType::Addition:
		return "Addition";
	case NodeType::BatchNormalization:
		return "BatchNormalization";
	case NodeType::Convolution:
		return "Convolution";
	case NodeType::Dense:
		return "Dense";
	case NodeType::Dropout:
		return "Dropout";
	case NodeType::ELU:
		return "ELU";
	case NodeType::Flatten:
		return "Flatten";
	case NodeType::LeakyReLU:
		return "LeakyReLU";
	case NodeType::Log:
		return "Log";
	case NodeType::MatrixMultiply:
		return "MatrixMultiply";
	case NodeType::Multiply:
		return "Multiply";
	case NodeType::Negate:
		return "Negate";
	case NodeType::None:
		return "None";
	case NodeType::Operation:
		return "Operation";
	case NodeType::Placeholder:
		return "Placeholder";
	case NodeType::Pooling:
		return "Pooling";
	case NodeType::ReduceSum:
		return "ReduceSum";
	case NodeType::ReLU:
		return "ReLU";
	case NodeType::Reshape:
		return "Reshape";
	case NodeType::SELU:
		return "SELU";
	case NodeType::SGD:
		return "SGD";
	case NodeType::Sigmoid:
		return "Sigmoid";
	case NodeType::Softmax:
		return "Softmax";
	case NodeType::SoftPlus:
		return "SoftPlus";
	case NodeType::SoftSign:
		return "SoftSign";
	case NodeType::TanH:
		return "TanH";
	case NodeType::Value:
		return "Value";
	default:
		return "Unknown";
	}
}

auto Node::is_operation(NodeType type) -> bool
{
	return IsOperationInTypelist<OperationTypes>::check(type);
}

}  // namespace Core