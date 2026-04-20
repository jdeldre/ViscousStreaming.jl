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

# need to use this version because I recalled a bug in original vdu, maybe I was wrong
# v.∇u, where v and u are both primal edge data
function _unscaled_convective_derivative!(vdu::Edges{Primal},v::Edges{Primal},u::Edges{Primal},extra_cache::ConvectiveDerivativeCache)
    @unpack vt1_cache, vt2_cache, vt3_cache = extra_cache

    fill!(vt1_cache,0.0)
    grid_interpolate!(vt1_cache,v)
    CartesianGrids.transpose!(vt2_cache,vt1_cache)
    fill!(vt1_cache,0.0)
    grad!(vt1_cache,u)
    product!(vt3_cache,vt2_cache,vt1_cache)
    fill!(vdu,0.0)
    grid_interpolate!(vdu,vt3_cache)
end

@ilmproblem ViscousStreamingInertial vector

struct ViscousStreamingInertialCache{SMT,SSMT,CMT,RCT,DVT,VNT,ST,VFT,FT, HT} <: AbstractExtraILMCache
   S1 :: SMT
   S2 :: SMT
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

function ImmersedLayers.prob_cache(prob::ViscousStreamingInertialProblem,base_cache::BasicILMCache)
    # Build Helmholtz operator with Re from phys_params
    Re = prob.phys_params["Re"]
    H = _get_helmholtz(1.0, Re, base_cache.g, true, GridScaling)

    S1 = create_CHLinvCT(base_cache, H)
    S2 = create_CL2invCT(base_cache)
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

    ViscousStreamingInertialCache(S1,S2,Ss,C,Rc,dv,vb,vprime,dvn,sstar,vϕ,ϕ,H)
end

function ImmersedLayers.solve(prob::ViscousStreamingInertialProblem,sys::ILMSystem)
    @unpack extra_cache, base_cache, bc, phys_params = sys
    @unpack nrm = base_cache
    @unpack S1, S2, Ss, C, Rc, dv, vb, vprime, sstar, dvn, vϕ, ϕ, H  = extra_cache

    Re = phys_params["Re"]

    σ1 = zeros_surface(sys)
    s1 = zeros_gridcurl(sys)
    v1 = zeros_grid(sys)

    # Get the jumps in velocity across surface
    prescribed_surface_jump!(dv,sys)

    # Compute r1
    surface_divergence_symm!(v1,dv,sys)
    curl!(sstar,v1,sys)

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
    σ1 .= S1\vprime

    # Correction streamfunction
    surface_curl!(s1,σ1,sys)
    s1 .*= -1.0
    
    s1 .= H \ s1
    inverse_laplacian!(s1,sys)

    # Correct
    s1 .+= sstar

    # Assemble the velocity
    curl!(v1,s1,sys)
    v1 .+= vϕ

    # Add the streamfunction equivalent to scalar potential
    ds = zeros_surfacescalar(base_cache)
    surface_grad_cross!(ds,ϕ,base_cache)
    ds .= Ss\ds
    surface_curl_cross!(sstar,ds,base_cache)
    sstar .*= -1.0
    inverse_laplacian!(sstar,base_cache)
    s1 .+= sstar

    # Filter the traction twice to clean it up a bit
    #σ .= C^2*σ

    ω1 = similar(s1)
    curl!(ω1,v1,sys)

    # compute jump in vorticity
    tract = zeros_surface(sys);
    vb_tmp = zeros_surface(sys);
    s_tmp = zeros_surfacescalar(sys);
    s_tmp2 = zeros_surfacescalar(sys);
    gam_tmp = zeros_surfacescalar(sys);

    dv_tmp = zeros_surface(sys);
    tract_field = zeros_grid(sys);

    gam_field = zeros_gridcurl(sys);

    prescribed_surface_average!(vb_tmp,sys);
    nrm = normals(sys);

    pointwise_dot!(s_tmp,nrm,vb_tmp);
    prescribed_surface_jump!(dv_tmp,sys);
    product!(tract,dv_tmp,conj(s_tmp));

    pointwise_cross!(gam_tmp,nrm,dv_tmp);
    ImmersedLayers.regularize!(gam_field,gam_tmp,sys);

    tract .*= -1/2;

    ImmersedLayers.regularize!(tract_field,tract,sys);

    # compute nonlinear convective term
    vdv2 = zeros_grid(sys);
    vdw2 = zeros_gridcurl(sys)

    cdrotcache = RotConvectiveDerivativeCache(sys.base_cache);
    w_cross_v!(vdv2,conj(ω1 - gam_field),v1,sys.base_cache,cdrotcache);
    vdv2 .*= 0.5;

    vdv2 .+= tract_field;

    curl!(vdw2,vdv2,sys);

    # drift velocity
    ud = zeros_grid(sys)
    cdcache = ConvectiveDerivativeCache(sys.base_cache)
    _unscaled_convective_derivative!(ud, conj(v1), v1, cdcache)
    ImmersedLayers._scale_derivative!(ud, sys.base_cache);
    ud .= 0.5 * real(im * ud);

    fb1 = zeros_surface(sys)
    ImmersedLayers.interpolate!(fb1, -ud, sys);
    fb1 .*= 2.0;

    # 2nd order solution
    σ2 = zeros_surface(sys)
    s2 = zeros_gridcurl(sys)
    v2 = zeros_grid(sys)

    # Get the jumps in velocity across surface
    dv .= fb1 - fb1

    # Compute ψ*
    sstar .= 0
    surface_divergence_symm!(v2,dv,sys)
    curl!(sstar,v2,sys)
    sstar .*= -1.0

    sstar .-= Re * real(vdw2)

    inverse_laplacian!(sstar,sys)
    inverse_laplacian!(sstar,sys)

    # Adjustment for jump in normal velocity
    vϕ .= 0
    ϕ .= 0
    dvn .= 0
    pointwise_dot!(dvn,nrm,dv)
    regularize!(ϕ,dvn,Rc)
    inverse_laplacian!(ϕ,sys)
    grad!(vϕ,ϕ,sys)

    # Get the average velocity on the surface
    vb .= 0.5 * (fb1 + fb1)
    vprime .= 0
    ImmersedLayers.interpolate!(vprime,vϕ,sys)
    vprime .= vb - vprime
    # reference frame correction
    # vprime.u .+= 1


    # Compute surface velocity due to ψ*
    surface_curl!(vb,sstar,sys)

    # Spurious slip
    vprime .-= vb
    σ2 .= S2\vprime

    # Correction streamfunction
    surface_curl!(s2,σ2,sys)
    s2 .*= -1.0
    inverse_laplacian!(s2,sys)
    inverse_laplacian!(s2,sys)

    # Correct
    s2 .+= sstar

    # reference frame correction
    # s .-= y_gridcurl(sys)

    # Assemble the velocity
    curl!(v2,s2,sys)
    v2 .+= vϕ

    # Add the streamfunction equivalent to scalar potential
    
    ds = zeros_surfacescalar(base_cache)
    surface_grad_cross!(ds,ϕ,base_cache)
    ds .= Ss\ds
    surface_curl_cross!(sstar,ds,base_cache)
    sstar .*= -1.0
    inverse_laplacian!(sstar,base_cache)
    s2 .+= sstar # should be +
    
    # Filter the traction twice to clean it up a bit
    σ2 .= C^2*σ2

    return v1, s1, σ1, v2, s2, σ2
end

@ilmproblem ViscousStreamingNonInertial vector

struct ViscousStreamingNonInertialCache{SMT,SSMT,CMT,RCT,DVT,VNT,ST,VFT,FT, HT} <: AbstractExtraILMCache
   S1 :: SMT
   S2 :: SMT
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

function ImmersedLayers.prob_cache(prob::ViscousStreamingNonInertialProblem,base_cache::BasicILMCache)
    # Build Helmholtz operator with Re from phys_params
    Re = prob.phys_params["Re"]
    H = _get_helmholtz(1.0, Re, base_cache.g, true, GridScaling)

    S1 = create_CHLinvCT(base_cache, H)
    S2 = create_CL2invCT(base_cache)
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

    ViscousStreamingNonInertialCache(S1,S2,Ss,C,Rc,dv,vb,vprime,dvn,sstar,vϕ,ϕ,H)
end

function ImmersedLayers.solve(prob::ViscousStreamingNonInertialProblem,sys::ILMSystem)
    @unpack extra_cache, base_cache, bc, phys_params = sys
    # @unpack extra_cache, base_cache, phys_params = sys
    # could unpack bc if we wanted to do something with it, but for now we just need phys_params for Re
    @unpack nrm = base_cache
    @unpack S1, S2, Ss, C, Rc, dv, vb, vprime, sstar, dvn, vϕ, ϕ, H  = extra_cache

    Re = phys_params["Re"]

    σ1 = zeros_surface(sys)
    s1 = zeros_gridcurl(sys)
    v1 = zeros_grid(sys)

    # Get the jumps in velocity across surface
    prescribed_surface_jump!(dv,sys)

    # Compute r1
    surface_divergence_symm!(v1,dv,sys)
    curl!(sstar,v1,sys)

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
    # reference frame correction
    vprime.u .+= 1

    # Compute surface velocity due to ψ*
    surface_curl!(vb, sstar, sys)

    # Spurious slip
    vprime .-= vb
    σ1 .= S1\vprime

    # Correction streamfunction
    surface_curl!(s1,σ1,sys)
    s1 .*= -1.0
    
    s1 .= H \ s1
    inverse_laplacian!(s1,sys)

    # Correct
    s1 .+= sstar

    # reference frame correction
    s1 .-= y_gridcurl(sys)

    # Assemble the velocity
    curl!(v1,s1,sys)
    v1 .+= vϕ

    # Add the streamfunction equivalent to scalar potential
    ds = zeros_surfacescalar(base_cache)
    surface_grad_cross!(ds,ϕ,base_cache)
    ds .= Ss\ds
    surface_curl_cross!(sstar,ds,base_cache)
    sstar .*= -1.0
    inverse_laplacian!(sstar,base_cache)
    s1 .+= sstar

    # Filter the traction twice to clean it up a bit
    #σ .= C^2*σ

    ω1 = similar(s1)
    curl!(ω1,v1,sys)

    # compute jump in vorticity
    tract = zeros_surface(sys);
    vb_tmp = zeros_surface(sys);
    s_tmp = zeros_surfacescalar(sys);
    s_tmp2 = zeros_surfacescalar(sys);
    gam_tmp = zeros_surfacescalar(sys);

    dv_tmp = zeros_surface(sys);
    tract_field = zeros_grid(sys);

    gam_field = zeros_gridcurl(sys);

    prescribed_surface_average!(vb_tmp,sys);
    nrm = normals(sys);

    pointwise_dot!(s_tmp,nrm,vb_tmp);
    prescribed_surface_jump!(dv_tmp,sys);
    product!(tract,dv_tmp,conj(s_tmp));

    pointwise_cross!(gam_tmp,nrm,dv_tmp);
    ImmersedLayers.regularize!(gam_field,gam_tmp,sys);

    tract .*= -1/2;

    ImmersedLayers.regularize!(tract_field,tract,sys);

    # compute nonlinear convective term
    vdv2 = zeros_grid(sys);
    vdw2 = zeros_gridcurl(sys)

    cdrotcache = RotConvectiveDerivativeCache(sys.base_cache);
    w_cross_v!(vdv2,conj(ω1 - gam_field),v1,sys.base_cache,cdrotcache);
    vdv2 .*= 0.5;

    vdv2 .+= tract_field;

    curl!(vdw2,vdv2,sys);

    # drift velocity
    ud = zeros_grid(sys)
    cdcache = ConvectiveDerivativeCache(sys.base_cache)
    _unscaled_convective_derivative!(ud, conj(v1), v1, cdcache)
    ImmersedLayers._scale_derivative!(ud, sys.base_cache);
    ud .= 0.5 * real(im * ud);

    fb1 = zeros_surface(sys)
    ImmersedLayers.interpolate!(fb1, -ud, sys);
    fb1 .*= 2.0;

    # 2nd order solution
    σ2 = zeros_surface(sys)
    s2 = zeros_gridcurl(sys)
    v2 = zeros_grid(sys)

    # Get the jumps in velocity across surface
    dv .= fb1 - fb1

    # Compute ψ*
    sstar .= 0
    surface_divergence_symm!(v2,dv,sys)
    curl!(sstar,v2,sys)
    sstar .*= -1.0

    sstar .-= Re * real(vdw2)

    inverse_laplacian!(sstar,sys)
    inverse_laplacian!(sstar,sys)

    # Adjustment for jump in normal velocity
    vϕ .= 0
    ϕ .= 0
    dvn .= 0
    pointwise_dot!(dvn,nrm,dv)
    regularize!(ϕ,dvn,Rc)
    inverse_laplacian!(ϕ,sys)
    grad!(vϕ,ϕ,sys)

    # Get the average velocity on the surface
    vb .= 0.5 * (fb1 + fb1)
    vprime .= 0
    ImmersedLayers.interpolate!(vprime,vϕ,sys)
    vprime .= vb - vprime


    # Compute surface velocity due to ψ*
    surface_curl!(vb,sstar,sys)

    # Spurious slip
    vprime .-= vb
    σ2 .= S2\vprime

    # Correction streamfunction
    surface_curl!(s2,σ2,sys)
    s2 .*= -1.0
    inverse_laplacian!(s2,sys)
    inverse_laplacian!(s2,sys)

    # Correct
    s2 .+= sstar



    # Assemble the velocity
    curl!(v2,s2,sys)
    v2 .+= vϕ

    # Add the streamfunction equivalent to scalar potential
    
    ds = zeros_surfacescalar(base_cache)
    surface_grad_cross!(ds,ϕ,base_cache)
    ds .= Ss\ds
    surface_curl_cross!(sstar,ds,base_cache)
    sstar .*= -1.0
    inverse_laplacian!(sstar,base_cache)
    s2 .+= sstar # should be +
    
    # Filter the traction twice to clean it up a bit
    σ2 .= C^2*σ2

    return v1, s1, σ1, v2, s2, σ2
end