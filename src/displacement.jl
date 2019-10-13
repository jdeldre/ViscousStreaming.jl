# routines for computing the disturbed displacement field

using Interpolations

export interpolated_history, compute_displacement_field, stokes_drift

function stokes_drift(vx::History{S,H},vy::History{T,H},trange,Δt::Float64,g::PhysicalGrid) where {S,T,H<:PeriodicHistory}

    ξi = Edges(Primal,vx[1])

    ξ_hist = compute_displacement_field(vx,vy,ξi,Δt,trange)

    v1tmp = Edges(Primal,ξi)
    v2tmp = Edges(Primal,ξi)

    vd = History(ξi,htype=PeriodicHistory)
    for (i,ξ) in enumerate(ξ_hist)
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
  length(tr) == length(v)+1 || error("Incommensurate lengths of history and range")
  return CubicSplineInterpolation(tr,Base.unsafe_view(v,1:length(tr)),extrapolation_bc=Periodic())
end

function interpolated_history(v::History{T,RegularHistory},tr::AbstractRange) where {T}
  length(tr) == length(v) || error("Incommensurate lengths of history and range")
  return CubicSplineInterpolation(tr,v,extrapolation_bc=Periodic())
end

"""
    interpolated_history(v::History,xr,yr,tr)

Returns a function that interpolates the history `v` both spatially and temporally,
using the x and y ranges specified by `xr` and `yr` and time specified by time range
`tr`. Uses cubic spline interpolation in all directions. If `v` is a periodic history,
then the resulting function is also periodic in time.
"""
function interpolated_history(v::History{T,PeriodicHistory},xr::AbstractRange,yr::AbstractRange,tr::AbstractRange) where {T}
  # v is a PeriodicHistory, so need to add first element to last here
  nx, ny = size(v[1])
  vhist = zeros(nx,ny,length(v)+1)
  for i in 1:size(vhist,3)
    vhist[:,:,i] = v[i]
  end
  (length(xr) == nx && length(yr) == ny && length(tr) == length(v)+1) || error("Incommensurate lengths of history and range")

  return CubicSplineInterpolation((xr, yr ,tr), vhist, extrapolation_bc = (Flat(),Flat(),Periodic()))

end


#=
Need to set up and perform the solution for displacement history
=#

function compute_displacement_field(vx::History{S,H},vy::History{T,H},ξi::Edges,Δt,trange) where {S,T,H <: PeriodicHistory}

  vxt = interpolated_history(vx,trange)
  vyt = interpolated_history(vy,trange)

  displacement_r₁(ξ::T,t::Float64) where {T <: VectorGridData} = T(vxt(t),vyt(t))

  solver = RK(ξi, Δt, displacement_r₁, rk=TimeMarching.RK31)

  t = 0.0
  t_hist = History(0.0,htype=PeriodicHistory)
  ξ_hist = History(ξi,htype=PeriodicHistory)
  u = deepcopy(ξi)
  for ti in trange
    push!(ξ_hist,deepcopy(u))
    push!(t_hist,copy(t))
    t,u = solver(t,u)
  end

  # remove the mean
  ξavg = mean(ξ_hist)
  ξ_hist .= map(ξ -> ξ - ξavg,ξ_hist)

  return ξ_hist

end
