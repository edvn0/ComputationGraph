#include "nodes/optimizers/SGDOptimizer.hpp"

namespace Core
{

    auto SGDOptimizer::forward(const std::vector<arma::mat> &consumer_outputs)
    -> void
    {
        auto shared_from = shared_from_this();
        auto computed_gradient = compute_gradients(shared_from);
    }

}  // namespace Core