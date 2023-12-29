#pragma once

#include <functional>
#include <utility>

namespace Core
{

template <class F> struct recursive
{
	F f;
	template <class... Ts> decltype(auto) operator()(Ts &&...ts) const
	{
		return f(std::ref(*this), std::forward<Ts>(ts)...);
	}

	template <class... Ts> decltype(auto) operator()(Ts &&...ts)
	{
		return f(std::ref(*this), std::forward<Ts>(ts)...);
	}
};

template <class F> recursive(F) -> recursive<F>;
auto const rec = [](auto f) { return recursive{std::move(f)}; };

}  // namespace Core