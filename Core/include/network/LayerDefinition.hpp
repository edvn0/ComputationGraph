#pragma once

#include "nodes/Node.hpp"

#include <cstdint>

namespace Core
{

struct LayerDefinition
{
	int units;
	NodeType activation;

	LayerDefinition(int u, NodeType a = NodeType::None)
		: units(u), activation(a)
	{
	}
};

}  // namespace Core
