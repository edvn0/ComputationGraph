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
	case NodeType::Addition:
		return "Addition";
	case NodeType::MatrixMultiply:
		return "MatrixMultiply";
	case NodeType::Negate:
		return "Negate";
	case NodeType::ReLU:
		return "ReLU";
	case NodeType::Sigmoid:
		return "Sigmoid";
	case NodeType::Softmax:
		return "Softmax";
	case NodeType::Placeholder:
		return "Placeholder";
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