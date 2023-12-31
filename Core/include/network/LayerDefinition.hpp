#pragma once

#include "nodes/Node.hpp"

#include <cstdint>

namespace Core
{

struct LayerDefinition
{
	u32 units{};
	NodeType activation;

	LayerDefinition(u32 neurons, NodeType type = NodeType::None)
		: units(neurons), activation(type)
	{
	}
};

}  // namespace Core
