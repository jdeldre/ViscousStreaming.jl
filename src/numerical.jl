using CartesianGrids

_get_helmholtz(coeff_factor::Real,α::Real,g::PhysicalGrid,with_inverse,::Type{IndexScaling}) =
               CartesianGrids.plan_helmholtz(g,with_inverse=with_inverse,factor=coeff_factor)
_get_helmholtz(coeff_factor::Real,α::Real,g::PhysicalGrid,with_inverse,::Type{GridScaling}) =
               CartesianGrids.plan_helmholtz(g,α*cellsize(g)^2,with_inverse=with_inverse,factor=coeff_factor/cellsize(g)^2)

function create_CHLinvCT(cache::BasicILMCache{N}, H;scale=1.0) where {N}
    @unpack L, sdata_cache, gdata_cache, gcurl_cache = cache

    len = length(sdata_cache)
    A = Matrix{eltype(sdata_cache)}(undef,len,len)
    fill!(sdata_cache,0.0)

    for col in 1:len
        sdata_cache[col] = 1.0
        fill!(gdata_cache,0.0)
        fill!(gcurl_cache,0.0)

        surface_curl!(gcurl_cache,sdata_cache,cache)
        gcurl_cache .= H \ gcurl_cache
        inverse_laplacian!(gcurl_cache,cache)
        
        surface_curl!(sdata_cache,gcurl_cache,cache)

        A[:,col] = scale*sdata_cache
        fill!(sdata_cache,0.0)
    end

    return -A
end

@ilmproblem ViscousStreaming vector

struct ViscousStreamingCache{SMT,SSMT,CMT,RCT,DVT,VNT,ST,VFT,FT, HT} <: AbstractExtraILMCache
   S :: SMT
   Ss :: SSMT
   C :: CMT
   Rc :: RCT
   dv :: DVT
   vb :: DVT
   vprime :: DVT
   dvn :: VNT
   sstar :: ST
   vϕ :: VFT
   ϕ :: FT
   H :: HT # Helmholtz operator
end

function ImmersedLayers.prob_cache(prob::ViscousStreamingProblem,base_cache::BasicILMCache)
    # Build Helmholtz operator with Re from phys_params
    Re = prob.phys_params["Re"]
    H = _get_helmholtz(1.0, Re, base_cache.g, true, GridScaling)

    S = create_CHLinvCT(base_cache, H)
    Ss = create_CLinvCT_scalar(base_cache)
    C = create_surface_filter(base_cache)

    dv = zeros_surface(base_cache)
    vb = zeros_surface(base_cache)
    vprime = zeros_surface(base_cache)

    dvn = ScalarData(dv)
    sstar = zeros_gridcurl(base_cache)
    vϕ = zeros_grid(base_cache)
    ϕ = Nodes(Primal,sstar)

    Rc = RegularizationMatrix(base_cache,dvn,ϕ)



    ViscousStreamingCache(S,Ss,C,Rc,dv,vb,vprime,dvn,sstar,vϕ,ϕ,H)
end

function ImmersedLayers.solve(prob::ViscousStreamingProblem,sys::ILMSystem)
    @unpack extra_cache, base_cache, bc, phys_params = sys
    @unpack nrm = base_cache
    @unpack S, Ss, C, Rc, dv, vb, vprime, sstar, dvn, vϕ, ϕ, H  = extra_cache

    σ = zeros_surface(sys)
    s = zeros_gridcurl(sys)
    v = zeros_grid(sys)

    # Get the jumps in velocity across surface
    prescribed_surface_jump!(dv,sys)

    # Compute r1
    surface_divergence_symm!(v,dv,sys)
    curl!(sstar,v,sys)

    # Compute ψ*
    sstar .= H \ sstar
    inverse_laplacian!(sstar,sys) 

    # Adjustment for jump in normal velocity
    pointwise_dot!(dvn,nrm,dv)
    regularize!(ϕ,dvn,Rc)
    inverse_laplacian!(ϕ,sys)
    grad!(vϕ,ϕ,sys)

    # Get the average velocity on the surface
    prescribed_surface_average!(vb,sys)
    ImmersedLayers.interpolate!(vprime,vϕ,sys)
    vprime .= vb - vprime # this is r2

    # Compute surface velocity due to ψ*
    surface_curl!(vb, sstar, sys)

    # Spurious slip
    vprime .-= vb
    σ .= S\vprime

    # Correction streamfunction
    surface_curl!(s,σ,sys)
    s .*= -1.0
    
    s .= H \ s
    inverse_laplacian!(s,sys)

    # Correct
    s .+= sstar

    # Assemble the velocity
    curl!(v,s,sys)
    v .+= vϕ

    # Add the streamfunction equivalent to scalar potential
    ds = zeros_surfacescalar(base_cache)
    surface_grad_cross!(ds,ϕ,base_cache)
    ds .= Ss\ds
    surface_curl_cross!(sstar,ds,base_cache)
    sstar .*= -1.0
    inverse_laplacian!(sstar,base_cache)
    s .+= sstar

    # Filter the traction twice to clean it up a bit
    #σ .= C^2*σ

    return v, s, σ, sstar, vϕ
end