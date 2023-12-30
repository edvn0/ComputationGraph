#pragma once

#include "nodes/OperationNode.hpp"

namespace Core {

    auto compute_gradients(Ref<Node>& root)-> std::unordered_map<Ref<Node>, arma::mat>;

    class SGDOptimizer : public OperationNode
    {
    public:
        ~SGDOptimizer() override = default;
        explicit SGDOptimizer(const std::vector<Ref<Node>> &inputs,
                                    double lr = 0.0001)
                : OperationNode(inputs), learning_rate(lr)
        {
        }
        auto forward(const std::vector<arma::mat> &consumer_outputs)
        -> void override;

        auto backward() -> void override
        {
        }

        auto get_type() const -> NodeType override
        {
            throw;
        }
        static auto get_operation_type() -> NodeType
        {
            throw;
        }
    private:
        [[maybe_unused]] double learning_rate {0.001};
    };

}