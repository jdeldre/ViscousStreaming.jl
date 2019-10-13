# routines for computing the disturbed displacement field

using Interpolations

export interpolated_history

"""
    interpolated_history(v::History,tr)

Returns a function that interpolates the history `v`, using the time specified
by time range `tr`. Uses cubic spline interpolation. If `v` is a periodic history,
then the resulting function is also periodic.
"""
function interpolated_history(v::History{T,PeriodicHistory},tr::AbstractRange) where {T}
  return CubicSplineInterpolation(tr,Base.unsafe_view(v,1:length(tr)),extrapolation_bc=Periodic())
end

function interpolated_history(v::History{T,RegularHistory},tr::AbstractRange) where {T}
  length(tr) == length(v) || error("Incommensurate lengths of history and range")
  return CubicSplineInterpolation(tr,v,extrapolation_bc=Periodic())
end
