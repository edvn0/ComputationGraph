#pragma once

#include "core/Typelist.hpp"
#include "nodes/operations/AdditionOperation.hpp"
#include "nodes/operations/LogOperation.hpp"
#include "nodes/operations/MatrixMultiplyOperation.hpp"
#include "nodes/operations/MultiplyOperation.hpp"
#include "nodes/operations/NegateOperation.hpp"
#include "nodes/operations/ReLUOperation.hpp"
#include "nodes/operations/ReduceSumOperation.hpp"
#include "nodes/operations/SoftmaxOperation.hpp"
#include "nodes/optimizers/SGDOptimizer.hpp"

namespace Core
{

using OperationTypes =
	Typelist<SoftmaxOperation, NegateOperation, ReLUOperation,
			 MultiplyOperation, ReduceSumOperation, AdditionOperation,
			 MatrixMultiplyOperation, LogOperation, SGDOptimizer>;

}  // namespace Core
