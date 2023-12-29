#pragma once

#include "nodes/Node.hpp"

#include <cstdint>

namespace Core
{

struct LayerDefinition
{
	std::uint32_t units{};
	NodeType activation;

	LayerDefinition(std::uint32_t neurons, NodeType type = NodeType::None)
		: units(neurons), activation(type)
	{
	}
};

}  // namespace Core
