# routines for computing the disturbed displacement field

using Interpolations

export interpolated_history, compute_displacement_field, stokes_drift

function stokes_drift(vx::History{S,H},vy::History{T,H},trange) where {S,T,H<:PeriodicHistory}

    ξi = Edges(Primal,vx[1])

    ξ_hist = compute_displacement_field(vx,vy,ξi,Δt,trange)

    v1tmp = Edges(Primal,ξi)
    v2tmp = Edges(Primal,ξi)

    vd = History(ξi,htype=PeriodicHistory)
    for (i,ξ) in enumerate(ξ_hist)
      ti = thist[i]
      directional_derivative!(v1tmp,typeof(ξi)(vx[i],vy[i]),ξ)
      directional_derivative!(v2tmp,ξ,typeof(ξi)(vx[i],vy[i]))
      v1tmp -= v2tmp
      v1tmp *= 0.5
      v1tmp /= cellsize(g)
      push!(vd,deepcopy(v1tmp))
    end

    return mean(vd)

end


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


#=
Need to set up and perform the solution for displacement history
=#

function compute_displacement_field(vx::History{S,H},vy::History{T,H},ξi::Edges,Δt,trange) where {S,T,H <: PeriodicHistory}

  vxt = interpolated_history(vx,trange)
  vyt = interpolated_history(vy,trange)

  TimeMarching.r₁(ξ::T,t::Float64) where {T <: VectorGridData} = T(vxt(t),vyt(t))

  solver = RK(ξi, Δt, TimeMarching.r₁, rk=TimeMarching.RK31)

  t = 0.0
  t_hist = History(0.0,htype=PeriodicHistory)
  ξ_hist = History(ξ,htype=PeriodicHistory)
  u = deepcopy(ξi)
  for ti in trange
    push!(ξ_hist,deepcopy(u))
    push!(t_hist,copy(t))
    global t,u = solver(t,u)
  end

  ξavg = mean(ξ_hist)
  ξ_hist .= map(ξ -> ξ - ξavg,ξ_hist)

  return ξ_hist

end
