## Here, define routines that
#  -

using Statistics
import Statistics: mean

export reynolds_decomposition!

"""
    reynolds_decomposition!(q̄::Edges,q̃x::Vector{XEdges},q̃y::Vector{YEdges},itr)

Perform a Reynolds decomposition of the vectors of velocity component histories `q̃x` and `q̃y`. Both
components of the mean are placed in `q̄`, and the original vectors of component histories are
overwritten with the fluctuations from the mean. The mean is computed over the range `itr`.
"""
function reynolds_decomposition!(q̄::Edges,q̃x::History{S},q̃y::History{T},itr) where {S<:XEdges, T<:YEdges}
    q̄.u .= mean(q̃x[itr])
    q̄.v .= mean(q̃y[itr]);
    q̃x .= map(q -> q - q̄.u,q̃x)
    q̃y .= map(q -> q - q̄.v,q̃y)

    return q̄, q̃x, q̃y
end
