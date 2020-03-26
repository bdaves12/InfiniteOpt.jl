# Define symbol inputs for different variable types
const Parameter = :Parameter

# Extend Base.copy for infinite parameters
function Base.copy(p::ParameterRef, new_model::InfiniteModel)::ParameterRef
    return ParameterRef(new_model, p.index)
end

# Internal structure for building InfOptParameters
mutable struct _ParameterInfoExpr
    has_lb::Bool
    lower_bound::Any
    has_ub::Bool
    upper_bound::Any
    has_dist::Bool
    distribution::Any
    has_set::Bool
    set::Any
end

# Default constructor
function _ParameterInfoExpr(; lower_bound = NaN, upper_bound = NaN,
                            distribution = NaN, set = NaN)
    # isnan(::Expr) is not defined so we need to do !== NaN
    return _ParameterInfoExpr(lower_bound !== NaN, lower_bound,
                              upper_bound !== NaN, upper_bound,
                              distribution !== NaN, distribution,
                              set !== NaN, set)
end

# Internal function for use in processing valid key word arguments
function _is_set_keyword(kw::Expr)
    return kw.args[1] in [:set, :lower_bound, :upper_bound, :distribution]
end

# Extend to assist in building InfOptParameters
function JuMP._set_lower_bound_or_error(_error::Function,
                                        info::_ParameterInfoExpr, lower)
    info.has_lb && _error("Cannot specify parameter lower_bound twice")
    info.has_dist && _error("Cannot specify parameter lower_bound and " *
                            "distribution")
    info.has_set && _error("Cannot specify parameter lower_bound and set")
    info.has_lb = true
    info.lower_bound = lower
    return
end

# Extend to assist in building InfOptParameters
function JuMP._set_upper_bound_or_error(_error::Function,
                                        info::_ParameterInfoExpr, upper)
    info.has_ub && _error("Cannot specify parameter upper_bound twice")
    info.has_dist && _error("Cannot specify parameter upper_bound and " *
                            "distribution")
    info.has_set && _error("Cannot specify parameter upper_bound and set")
    info.has_ub = true
    info.upper_bound = upper
    return
end

# Extend to assist in building InfOptParameters
function _dist_or_error(_error::Function, info::_ParameterInfoExpr, dist)
    info.has_dist && _error("Cannot specify parameter distribution twice")
    (info.has_lb || info.has_ub) && _error("Cannot specify parameter " *
                                           "distribution and upper/lower bounds")
    info.has_set && _error("Cannot specify parameter distribution and set")
    info.has_dist = true
    info.distribution = dist
    return
end

# Extend to assist in building InfOptParameters
function _set_or_error(_error::Function, info::_ParameterInfoExpr, set)
    info.has_set && _error("Cannot specify variable fixed value twice")
    (info.has_lb || info.has_ub) && _error("Cannot specify parameter set and " *
                                           "upper/lower bounds")
    info.has_dist && _error("Cannot specify parameter set and distribution")
    info.has_set = true
    info.set = set
    return
end

# Construct an expression to build an infinite set (use with @infinite_macro)
function _constructor_set(_error::Function, info::_ParameterInfoExpr)
    if (info.has_lb || info.has_ub) && !(info.has_lb && info.has_ub)
        _error("Must specify both an upper bound and a lower bound")
    elseif info.has_lb
        check = :(isa($(info.lower_bound), Number))
        return :($(check) ? IntervalSet($(info.lower_bound), $(info.upper_bound)) : error("Bounds must be a number."))
    elseif info.has_dist
        check = :(isa($(info.distribution), Distributions.NonMatrixDistribution))
        return :($(check) ? DistributionSet($(info.distribution)) : error("Distribution must be a subtype of Distributions.NonMatrixDistribution."))
    elseif info.has_set
        check1 = :(isa($(info.set), AbstractInfiniteSet))
        check2 = :(isa($(info.set), Distributions.NonMatrixDistribution))
        return :($(check1) ? $(info.set) : ($(check2) ? DistributionSet($(info.set)) : error("Set must be a subtype of AbstractInfiniteSet.")))
    else
        _error("Must specify upper/lower bounds, a distribution, or a set")
    end
end

# Check that supports don't violate the set bounds
function _check_supports_in_bounds(_error::Function,
                                   supports::Union{Number, Vector{<:Number}},
                                   set::AbstractInfiniteSet)
    if !supports_in_set(supports, set)
        _error("Supports violate the set domain bounds.")
    end
    return
end

"""
    build_parameter(_error::Function, set::AbstractInfiniteSet,
                    [num_params::Int = 1; num_supports::Int = 0,
                    supports::Union{Number, Vector{<:Number}} = Number[],
                    independent::Bool = false,
                    sig_fig::Int = 5])::InfOptParameter

Returns a [`InfOptParameter`](@ref) given the appropriate information. This is
analagous to `JuMP.build_variable`. Errors if supports violate the bounds
associated `set`. Also errors if `set` contains a multivariate distribution with
a different dimension than `num_params`. This is meant to primarily serve as a
helper method for [`@infinite_parameter`](@ref).

**Example**
```jldoctest; setup = :(using InfiniteOpt)
julia> build_parameter(error, IntervalSet(0, 3), supports = Vector(0:3))
InfOptParameter{IntervalSet}([0, 3], [0, 1, 2, 3], false)
```
"""
function build_parameter(_error::Function, set::AbstractInfiniteSet,
                         num_params::Int = 1;
                         num_supports::Int = 0,
                         supports::Union{Number, Vector{<:Number}} = Number[],
                         independent::Bool = false,
                         sig_fig::Int = 5,
                         extra_kw_args...)::InfOptParameter
    for (kwarg, _) in extra_kw_args
        _error("Unrecognized keyword argument $kwarg")
    end
    length_supports = length(supports)
    if num_supports == 0 && length_supports != 0
        _check_supports_in_bounds(_error, supports, set)
    elseif num_supports != 0 && length_supports != 0
        @warn("Ignoring num_supports since supports is not empty.")
        _check_supports_in_bounds(_error, supports, set)
    elseif num_supports != 0 && length_supports == 0
        if isa(set, DistributionSet{<:Distributions.MultivariateDistribution})
            _error("Support generation is not available for multivariate " *
                   "distributions.")
        end
        supports = generate_support_values(set, num_supports = num_supports,
                                           sig_fig = sig_fig)
    end
    if isa(set, DistributionSet{<:Distributions.MultivariateDistribution})
        if num_params != length(set.distribution)
            _error("Multivariate distribution dimension must match dimension " *
                   "of parameter.")
        end
    end
    if num_params == 1 || independent # double check num_params for other types of containers
        unique_supports = unique(supports)
        if length(unique_supports) != length(supports)
            @warn("Support points are not unique, eliminating redundant points.")
        end
        return InfOptParameter(set, unique_supports, independent)
    end
    if isa(supports, Number)
        supports = [supports]
    end
    return InfOptParameter(set, supports, independent)
end

# Check the number of supports of one dimension matches the other dimension
# within the same multi-dimensional parameter
function _check_supports_dimensions(model::InfiniteModel, p::InfOptParameter,
    index::Int)
    if haskey(model.params, index - 1) &&
       model.param_to_group_id[index - 1] == model.next_param_id
        if length(p.supports) != length(model.params[index - 1].supports)
            error("Support dimension mismatch. Make sure support dimension of " *
                  "each parameter element match.")
        end
    end
    return
end

"""
    add_parameter(model::InfiniteModel, p::InfOptParameter,
                  [name::String = ""])::ParameterRef

Returns a [`ParameterRef`](@ref) associated with the parameter `p` that is added
to `model`. This adds a parameter to the model in a manner similar to
`JuMP.add_variable`. This can be used to add parameters with the use of
[`@infinite_parameter`](@ref). [`build_parameter`](@ref) should be used to
construct `p`.

**Example**
```jldoctest; setup = :(using InfiniteOpt; model = InfiniteModel())
julia> p = build_parameter(error, IntervalSet(0, 3), supports = Vector(0:3))
InfOptParameter{IntervalSet}([0, 3], [0, 1, 2, 3], false)

julia> param_ref = add_parameter(model, p, "name")
name
```
"""
function add_parameter(model::InfiniteModel, p::InfOptParameter,
                       name::String=""; multi_dim = false,
                       macro_call = false)::ParameterRef
    index = model.next_param_index += 1
    pref = ParameterRef(model, index)
    if !macro_call
        model.next_param_id += 1
    elseif !(p.independent) && multi_dim
        _check_supports_dimensions(model, p, index)
    end
    model.params[JuMP.index(pref)] = p
    model.param_to_group_id[JuMP.index(pref)] = model.next_param_id
    JuMP.set_name(pref, name)
    return pref
end


"""
    used_by_constraint(pref::ParameterRef)::Bool

Return true if `pref` is used by a constraint or false otherwise.

**Example**
```julia-repl
julia> used_by_constraint(t)
true
```
"""
function used_by_constraint(pref::ParameterRef)::Bool
    return haskey(JuMP.owner_model(pref).param_to_constrs, JuMP.index(pref))
end

"""
    used_by_measure(pref::ParameterRef)::Bool

Return true if `pref` is used by a measure or false otherwise.

**Example**
```julia-repl
julia> used_by_measure(t)
false
```
"""
function used_by_measure(pref::ParameterRef)::Bool
    return haskey(JuMP.owner_model(pref).param_to_meas, JuMP.index(pref))
end

"""
    used_by_variable(pref::ParameterRef)::Bool

Return true if `pref` is used by an infinite variable or false otherwise.

**Example**
```julia-repl
julia> used_by_variable(t)
true
```
"""
function used_by_variable(pref::ParameterRef)::Bool
    return haskey(JuMP.owner_model(pref).param_to_vars, JuMP.index(pref))
end

"""
    is_used(pref::ParameterRef)::Bool

Return true if `pref` is used in the model or false otherwise.

**Example**
```julia-repl
julia> is_used(t)
true
```
"""
function is_used(pref::ParameterRef)::Bool
    return used_by_measure(pref) || used_by_constraint(pref) || used_by_variable(pref)
end

# Check if parameter is used by measure data and error if it is to prevent bad
# deleting behavior
function _check_param_in_data(pref::ParameterRef, data::AbstractMeasureData)
    prefs = parameter_refs(data)
    if (pref == prefs || pref in prefs)
        error("Unable to delete `$pref` since it is used to evaluate measures.")
    end
    return
end

## Determine if a tuple element contains a particular parameter
# ParameterRef
function _contains_pref(search_pref::ParameterRef, pref::ParameterRef)::Bool
    return search_pref == pref
end

# SparseAxisArray
function _contains_pref(arr::JuMPC.SparseAxisArray{<:ParameterRef},
                        pref::ParameterRef)::Bool
    return pref in collect(values(arr.data))
end

# Return parameter tuple without a particular parameter and return location of
# where it was
function _remove_parameter(prefs::Tuple, delete_pref::ParameterRef)::Tuple
    for i in eachindex(prefs)
        if _contains_pref(prefs[i], delete_pref)
            if isa(prefs[i], ParameterRef)
                return Tuple(prefs[j] for j = 1:length(prefs) if j != i), (i, )
            else
                for (k, v) in prefs[i].data
                    if v == delete_pref
                        new_dict = filter(x -> x.second != delete_pref,
                                         prefs[i].data)
                        if length(new_dict) != 0
                            pref_list = [prefs...]
                            pref_list[i] = JuMPC.SparseAxisArray(new_dict)
                            return Tuple(pref_list[j] for j = 1:length(prefs)), (i, k)
                        else
                            return Tuple(prefs[j] for j = 1:length(prefs) if j != i), (i, )
                        end
                    end
                end
            end
        end
    end
end

# Used to update infinite variable when one of its parameters is deleted
function _update_infinite_variable(vref::InfiniteVariableRef, new_prefs::Tuple)
    _update_variable_param_refs(vref, new_prefs)
    JuMP.set_name(vref, _root_name(vref))
    if used_by_measure(vref)
        for mindex in JuMP.owner_model(vref).var_to_meas[JuMP.index(vref)]
            JuMP.set_name(MeasureRef(JuMP.owner_model(vref), mindex),
                       _make_meas_name(JuMP.owner_model(vref).measures[mindex]))
        end
    end
    return
end

# Return a parameter value tuple without element at a particular location
function _remove_parameter_values(pref_vals::Tuple, location::Tuple)::Tuple
    # removed parameter was a scalar parameter
    if length(location) == 1
        return Tuple(pref_vals[i] for i in eachindex(pref_vals) if i != location[1])
    # removed parameter was part of an array
    else
        new_dict = filter(x -> x.first != location[2], pref_vals[location[1]].data)
        val_list = [pref_vals...]
        val_list[location[1]] = JuMPC.SparseAxisArray(new_dict)
        return Tuple(val_list[i] for i in eachindex(pref_vals))
    end
end

# Update point variable for which a parameter is deleted
function _update_point_variable(pvref::PointVariableRef, pref_vals::Tuple)
    _update_variable_param_values(pvref, pref_vals)
    # update name if no alias was provided
    if !isa(findfirst(isequal('('), JuMP.name(pvref)), Nothing)
        JuMP.set_name(pvref, "")
    end
    if used_by_measure(pvref)
        for mindex in JuMP.owner_model(pvref).var_to_meas[JuMP.index(pvref)]
            JuMP.set_name(MeasureRef(JuMP.owner_model(pvref), mindex),
                      _make_meas_name(JuMP.owner_model(pvref).measures[mindex]))
        end
    end
    return
end

# Update a reduced variable associated with an infinite variable whose parameter
# was removed
function _update_reduced_variable(vref::ReducedInfiniteVariableRef,
                                  location::Tuple)
    eval_supps = eval_supports(vref)
    # removed parameter was a scalar
    if length(location) == 1
        new_supports = Dict{Int, Union{Number, JuMPC.SparseAxisArray}}()
        for (index, support) in eval_supps
            if index < location[1]
                new_supports[index] = support
            elseif index > location[1]
                new_supports[index - 1] = support
            end
        end
        JuMP.owner_model(vref).reduced_variable[JuMP.index(vref)] = ReducedInfiniteVariable(infinite_variable_ref(vref), new_supports)
    # removed parameter was part of an array and was reduced previously
    elseif haskey(eval_supps, location[1])
        new_dict = filter(x -> x.first != location[2], eval_supps[location[1]].data)
        new_supports = copy(eval_supps)
        new_supports[location[1]] = JuMPC.SparseAxisArray(new_dict)
        JuMP.owner_model(vref).reduced_variable[JuMP.index(vref)] = ReducedInfiniteVariable(infinite_variable_ref(vref), new_supports)
    end
    if used_by_measure(vref)
        for mindex in JuMP.owner_model(vref).reduced_to_meas[JuMP.index(vref)]
            JuMP.set_name(MeasureRef(JuMP.owner_model(vref), mindex),
                       _make_meas_name(JuMP.owner_model(vref).measures[mindex]))
        end
    end
    return
end

"""
    JuMP.delete(model::InfiniteModel, pref::ParameterRef)

Extend [`JuMP.delete`](@ref JuMP.delete(::JuMP.Model, ::JuMP.VariableRef)) to delete
infinite parameters and their dependencies. All variables, constraints, and
measure functions that depend on `pref` are updated to exclude it. Errors if the
parameter is contained in an `AbstractMeasureData` datatype that is employed by
a measure since the measure becomes invalid otherwise. Thus, measures that
contain this dependency must be deleted first. Note that
```_check_param_in_data(pref, measure_data)``` needs to be extended to allow
deletion of parameters when custom `AbstractMeasureData` datatypes are used.

**Example**
```julia-repl
julia> print(model)
Min measure(g(t, x)*t + x) + z
Subject to
 z >= 0.0
 g(t, x) + z >= 42.0
 g(0.5, x) == 0
 t in [0, 6]
 x in [0, 1]

julia> delete(model, x)

julia> print(model)
Min measure(g(t)*t) + z
Subject to
 g(t) + z >= 42.0
 g(0.5) == 0
 z >= 0.0
 t in [0, 6]
```
"""
function JuMP.delete(model::InfiniteModel, pref::ParameterRef)
    @assert JuMP.is_valid(model, pref) "Parameter reference is invalid."
    # update optimizer model status
    if is_used(pref)
        set_optimizer_model_ready(model, false)
    end
    # update measures
    if used_by_measure(pref)
        # ensure deletion is okay (pref isn't used by measure data)
        for mindex in model.param_to_meas[JuMP.index(pref)]
            _check_param_in_data(pref, model.measures[mindex].data)
        end
        # delete dependence of measures on pref
        for mindex in model.param_to_meas[JuMP.index(pref)]
            if isa(model.measures[mindex].func, ParameterRef)
                model.measures[mindex] = Measure(zero(JuMP.AffExpr),
                                                 model.measures[mindex].data)
            else
                _remove_variable(model.measures[mindex].func, pref)
            end
            JuMP.set_name(MeasureRef(model, mindex),
                          _make_meas_name(model.measures[mindex]))
        end
        # delete mapping
        delete!(model.param_to_meas, JuMP.index(pref))
    end
    # update variables
    if used_by_variable(pref)
        # update infinite variables that depend on pref
        for vindex in model.param_to_vars[JuMP.index(pref)]
            # find location of parameter in storage tuple
            prefs, location = _remove_parameter(model.vars[vindex].parameter_refs,
                                                pref)
            vref = InfiniteVariableRef(model, vindex)
            # remove the parameter dependence
            _update_infinite_variable(vref, prefs)
            # update any point variables that depend on vref accordingly
            if used_by_point_variable(vref)
                for pindex in model.infinite_to_points[vindex]
                    pvref = PointVariableRef(model, pindex)
                    pref_vals = _remove_parameter_values(parameter_values(pvref),
                                                         location)
                    _update_point_variable(pvref, pref_vals)
                end
            end
            # update any reduced variables that depend on vref accordingly
            if used_by_reduced_variable(vref)
                for rindex in model.infinite_to_reduced[vindex]
                    _update_reduced_variable(ReducedInfiniteVariableRef(model,
                                                              rindex), location)
                end
            end
        end
        # delete mapping
        delete!(model.param_to_vars, JuMP.index(pref))
    end
    # update constraints
    if used_by_constraint(pref)
        # update constraints in mapping to remove the parameter
        for cindex in model.param_to_constrs[JuMP.index(pref)]
            if isa(model.constrs[cindex].func, ParameterRef)
                model.constrs[cindex] = JuMP.ScalarConstraint(zero(JuMP.AffExpr),
                                                      model.constrs[cindex].set)
            else
                _remove_variable(model.constrs[cindex].func, pref)
            end
        end
        # delete mapping
        delete!(model.param_to_constrs, JuMP.index(pref))
    end
    # delete parameter information stored in model
    delete!(model.params, JuMP.index(pref))
    delete!(model.param_to_name, JuMP.index(pref))
    delete!(model.param_to_group_id, JuMP.index(pref))
    return
end

"""
    JuMP.is_valid(model::InfiniteModel, pref::ParameterRef)::Bool

Extend the [`JuMP.is_valid`](@ref JuMP.is_valid(::JuMP.Model, ::JuMP.VariableRef))
function to accomodate infinite parameters.
Returns true if the `InfiniteModel` stored in `pref` matches `model` and if
the parameter index is used by `model`. It returns false otherwise.

**Example**
```jldoctest; setup = :(using InfiniteOpt, JuMP; model = InfiniteModel(); @infinite_parameter(model, t in [0, 1]))
julia> is_valid(model, t)
true
```
"""
function JuMP.is_valid(model::InfiniteModel, pref::ParameterRef)::Bool
    check1 = model === JuMP.owner_model(pref)
    check2 = JuMP.index(pref) in keys(model.params)
    return check1 && check2
end

"""
    JuMP.name(pref::ParameterRef)::String

Extend the [`JuMP.name`](@ref JuMP.name(::JuMP.VariableRef)) function to
accomodate infinite parameters. Returns the name string associated with `pref`.

**Example**
```jldoctest; setup = :(using InfiniteOpt, JuMP; model = InfiniteModel(); @infinite_parameter(model, t in [0, 1]))
julia> name(t)
"t"
```
"""
function JuMP.name(pref::ParameterRef)::String
    return JuMP.owner_model(pref).param_to_name[JuMP.index(pref)]
end

"""
    JuMP.set_name(pref::ParameterRef, name::String)

Extend the [`JuMP.set_name`](@ref JuMP.set_name(::JuMP.VariableRef, ::String))
function to accomodate infinite parameters. Set a new base name to be associated
with `pref`.

**Example**
```jldoctest; setup = :(using InfiniteOpt, JuMP; model = InfiniteModel(); @infinite_parameter(model, t in [0, 1]))
julia> set_name(t, "time")

julia> name(t)
"time"
```
"""
function JuMP.set_name(pref::ParameterRef, name::String)
    JuMP.owner_model(pref).param_to_name[JuMP.index(pref)] = name
    JuMP.owner_model(pref).name_to_param = nothing
    return
end

"""
    num_parameters(model::InfiniteModel)::Int

Return the number of infinite parameters currently present in `model`.

**Example**
```jldoctest; setup = :(using InfiniteOpt, JuMP; model = InfiniteModel(); @infinite_parameter(model, t in [0, 1]))
julia> num_parameters(model)
1
```
"""
function num_parameters(model::InfiniteModel)::Int
    return length(model.params)
end

# Internal functions
_parameter_set(pref::ParameterRef) = JuMP.owner_model(pref).params[JuMP.index(pref)].set
_parameter_supports(pref::ParameterRef) = JuMP.owner_model(pref).params[JuMP.index(pref)].supports
function _update_parameter_set(pref::ParameterRef, set::AbstractInfiniteSet)
    supports = JuMP.owner_model(pref).params[JuMP.index(pref)].supports
    independent = JuMP.owner_model(pref).params[JuMP.index(pref)].independent
    JuMP.owner_model(pref).params[JuMP.index(pref)] = InfOptParameter(set, supports, independent)
    if is_used(pref)
        set_optimizer_model_ready(JuMP.owner_model(pref), false)
    end
    return
end
function _update_parameter_supports(pref::ParameterRef, supports::Vector{<:Number})
    set = JuMP.owner_model(pref).params[JuMP.index(pref)].set
    independent = JuMP.owner_model(pref).params[JuMP.index(pref)].independent
    JuMP.owner_model(pref).params[JuMP.index(pref)] = InfOptParameter(set, supports, independent)
    if is_used(pref)
        set_optimizer_model_ready(JuMP.owner_model(pref), false)
    end
    return
end

"""
    infinite_set(pref::ParameterRef)::AbstractInfiniteSet

Return the infinite set associated with `pref`.

**Example**
```jldoctest; setup = :(using InfiniteOpt, JuMP; model = InfiniteModel(); @infinite_parameter(model, t in [0, 1]))
julia> infinite_set(t)
[0, 1]
```
"""
function infinite_set(pref::ParameterRef)::AbstractInfiniteSet
    return _parameter_set(pref)
end

"""
    set_infinite_set(pref::ParameterRef, set::AbstractInfiniteSet)

Specify the infinite set of `pref`.

**Example**
```jldoctest; setup = :(using InfiniteOpt, JuMP; model = InfiniteModel(); @infinite_parameter(model, t in [0, 1]))
julia> set_infinite_set(t, IntervalSet(0, 2))

julia> infinite_set(t)
[0, 2]
```
"""
function set_infinite_set(pref::ParameterRef, set::AbstractInfiniteSet)
    _update_parameter_set(pref, set)
    return
end

"""
    JuMP.has_lower_bound(pref::ParameterRef)::Bool

Extend the `JuMP.has_lower_bound` function to accomodate infinite parameters.
Return true if the set associated with `pref` has a defined lower bound or if a
lower bound can be found. Extensions with user-defined infinite set types
should extend `JuMP.has_lower_bound(set::NewType)`.

**Example**
```jldoctest; setup = :(using InfiniteOpt, JuMP; model = InfiniteModel(); @infinite_parameter(model, t in [0, 1]))
julia> has_lower_bound(t)
true
```
"""
function JuMP.has_lower_bound(pref::ParameterRef)::Bool
    set = _parameter_set(pref)
    return JuMP.has_lower_bound(set)
end

"""
    JuMP.lower_bound(pref::ParameterRef)::Number

Extend the `JuMP.lower_bound` function to accomodate infinite parameters.
Returns the lower bound associated with the infinite set. Errors if such a bound
is not well-defined.

**Example**
```jldoctest; setup = :(using InfiniteOpt, JuMP; model = InfiniteModel(); @infinite_parameter(model, t in [0, 1]))
julia> lower_bound(t)
0.0
```
"""
function JuMP.lower_bound(pref::ParameterRef)::Number
    set = _parameter_set(pref)
    if !JuMP.has_lower_bound(pref)
        error("Parameter $(pref) does not have a lower bound.")
    end
    return JuMP.lower_bound(set)
end

"""
    JuMP.set_lower_bound(pref::ParameterRef, lower::Number)

Extend the `JuMP.set_lower_bound` function to accomodate infinite parameters.
Updates the infinite set lower bound if such an operation is supported. Set
extensions that seek to employ this should extend
`JuMP.set_lower_bound(set::NewType, lower::Number)`.

**Example**
```jldoctest; setup = :(using InfiniteOpt, JuMP; model = InfiniteModel(); @infinite_parameter(model, t in [0, 1]))
julia> set_lower_bound(t, -1)

julia> lower_bound(t)
-1.0
```
"""
function JuMP.set_lower_bound(pref::ParameterRef, lower::Number)
    set = _parameter_set(pref)
    new_set = JuMP.set_lower_bound(set, lower)
    _update_parameter_set(pref, new_set)
    return
end

"""
    JuMP.has_upper_bound(pref::ParameterRef)::Bool

Extend the `JuMP.has_upper_bound` function to accomodate infinite parameters.
Return true if the set associated with `pref` has a defined upper bound or if a
upper bound can be found. Extensions with user-defined sets should extend
`JuMP.has_upper_bound(set::NewType)`.

**Example**
```jldoctest; setup = :(using InfiniteOpt, JuMP; model = InfiniteModel(); @infinite_parameter(model, t in [0, 1]))
julia> has_upper_bound(t)
true
```
"""
function JuMP.has_upper_bound(pref::ParameterRef)::Bool
    set = _parameter_set(pref)
    return JuMP.has_upper_bound(set)
end

"""
    JuMP.upper_bound(pref::ParameterRef)::Number

Extend the `JuMP.upper_bound` function to accomodate infinite parameters.
Returns the upper bound associated with the infinite set. Errors if such a bound
is not well-defined. Extensions with user-defined set types should extend
`JuMP.has_upper_bound(set::NewType)` and `JuMP.upper_bound(set::NewType)` if
appropriate.

**Example**
```jldoctest; setup = :(using InfiniteOpt, JuMP; model = InfiniteModel(); @infinite_parameter(model, t in [0, 1]))
julia> upper_bound(t)
1.0
```
"""
function JuMP.upper_bound(pref::ParameterRef)::Number
    set = _parameter_set(pref)
    if !JuMP.has_upper_bound(pref)
        error("Parameter $(pref) does not have a upper bound.")
    end
    return JuMP.upper_bound(set)
end

"""
    JuMP.set_upper_bound(pref::ParameterRef, lower::Number)

Extend the `JuMP.set_upper_bound` function to accomodate infinite parameters.
Updates the infinite set upper bound if and only if it is an IntervalSet. Errors
otherwise. Extensions with user-defined infinite sets should extend
`JuMP.set_upper_bound(set::NewType, upper::Number)` if appropriate.

**Example**
```jldoctest; setup = :(using InfiniteOpt, JuMP; model = InfiniteModel(); @infinite_parameter(model, t in [0, 1]))
julia> set_upper_bound(t, 2)

julia> upper_bound(t)
2.0
```
"""
function JuMP.set_upper_bound(pref::ParameterRef, upper::Number)
    set = _parameter_set(pref)
    new_set = JuMP.set_upper_bound(set, upper)
    _update_parameter_set(pref, new_set)
    return
end

"""
    num_supports(pref::ParameterRef)::Int

Return the number of support points associated with `pref`.

**Example**
```jldoctest; setup = :(using InfiniteOpt, JuMP; model = InfiniteModel(); @infinite_parameter(model, t in [0, 1], supports = [0, 1]))
julia> num_supports(t)
2
```
"""
function num_supports(pref::ParameterRef)::Int
    return length(_parameter_supports(pref))
end

"""
    has_supports(pref::ParameterRef)::Bool

Return true if `pref` has supports or false otherwise.

**Example**
```jldoctest; setup = :(using InfiniteOpt, JuMP; model = InfiniteModel(); @infinite_parameter(model, t in [0, 1], supports = [0, 1]))
julia> has_supports(t)
true
```
"""
has_supports(pref::ParameterRef)::Bool = num_supports(pref) > 0

"""
    supports(pref::ParameterRef)::Vector

Return the support points associated with `pref`. Errors if there are no
supports.

**Example**
```jldoctest; setup = :(using InfiniteOpt, JuMP; model = InfiniteModel(); @infinite_parameter(model, t in [0, 1], supports = [0, 1]))
julia> supports(t)
2-element Array{Int64,1}:
 0
 1
```
"""
function supports(pref::ParameterRef)::Vector
    !has_supports(pref) && error("Parameter $pref does not have supports.")
    return _parameter_supports(pref)
end

"""
    supports(prefs::AbstractArray{<:ParameterRef})::Vector

Return the support points associated with an array of `prefs` formatted as a
vector of SparseAxisArrays following the format of the input array. If the
parameters are not independent then the supports of each parameter are simply
spliced together. Alternatively can call `supports.` to more efficiently obtain
an array of the same input format whose parameter references have been replaced
with their supports. Errors if all the parameter references do not have the same
group ID number (were intialized together as an array) or if the nonindependent
parameters have support vectors of different lengths. If the parameters are
independent then all the unique combinations are identified and returned as
supports. Warning this operation is computationally expensive if there exist a
large number of combinations.

**Example**
```jldoctest; setup = :(using InfiniteOpt, JuMP; model = InfiniteModel())
julia> x = @infinite_parameter(model, [i = 1:2], set = IntervalSet(-1, 1),
                               base_name = "x", independent = true)
2-element Array{ParameterRef,1}:
 x[1]
 x[2]

julia> for i = 1:length(x)
           set_supports(x[i], [-1, 1])
       end

julia> supports(x)
4-element Array{JuMP.Containers.SparseAxisArray,1}:
   [2]  =  -1
  [1]  =  -1
   [2]  =  1
  [1]  =  -1
   [2]  =  -1
  [1]  =  1
   [2]  =  1
  [1]  =  1
```
"""
function supports(prefs::AbstractArray{<:ParameterRef})::Vector
    prefs = convert(JuMPC.SparseAxisArray, prefs)
    !_only_one_group(prefs) && error("Array contains parameters from multiple" *
                                     " groups.")
    for (k, pref) in prefs.data
        !has_supports(pref) && error("Parameter $pref does not have supports.")
    end
    lengths = [num_supports(pref) for (k, pref) in prefs.data]
    if !is_independent(collect(values(prefs.data))[1])
        length(unique(lengths)) != 1 && error("Each nonindependent parameter " *
                                              "must have the same number of " *
                                              "support points.")
        support_list = Vector{JuMPC.SparseAxisArray}(undef, lengths[1])
        for i in eachindex(support_list)
            support_list[i] = JuMPC.SparseAxisArray(Dict(k => supports(pref)[i] for (k, pref) in prefs.data))
        end
        # unique-nize support_list using the dictionary unique method
        # TODO: make it work with JuMPC.SparseAxisArray
        dict_list = unique([arr.data for arr in support_list])
        support_list = [JuMPC.SparseAxisArray(dict) for dict in dict_list]
    else
        all_keys = collect(keys(prefs))
        all_supports = [supports(pref) for (k, pref) in prefs.data]
        support_list = Vector{JuMPC.SparseAxisArray}(undef, prod(lengths))
        counter = 1
        for combo in Iterators.product(all_supports...)
            support_list[counter] = JuMPC.SparseAxisArray(Dict(all_keys[i] => combo[i] for i in eachindex(combo)))
            counter += 1
        end
    end
    return support_list
end

"""
    set_supports(pref::ParameterRef, supports::Vector{<:Number}; [force = false])

Specify the support points for `pref`. Errors if the supports violate the bounds
associated with the infinite set. Warns if the points are not unique. If `force`
this will overwrite exisiting supports otherwise it will error if there are
existing supports.

**Example**
```jldoctest; setup = :(using InfiniteOpt, JuMP; model = InfiniteModel(); @infinite_parameter(model, t in [0, 1]))
julia> set_supports(t, [0, 1])

julia> supports(t)
2-element Array{Int64,1}:
 0
 1
```
"""
function set_supports(pref::ParameterRef, supports::Vector{<:Number};
                      force = false)
    set = _parameter_set(pref)
    _check_supports_in_bounds(error, supports, set)
    if has_supports(pref) && !force
        error("Unable set supports for $pref since it already has supports." *
              " Consider using `add_supports` or use set `force = true` to " *
              "overwrite the existing supports.")
    end
    if !(is_independent(pref)) &&
       sum(values(pref.model.param_to_group_id) .== group_id(pref)) > 1
        _update_parameter_supports(pref, supports)
    else
        unique_supports = unique(supports)
        if length(unique_supports) != length(supports)
            @warn("Support points are not unique, eliminating redundant points.")
        end
        _update_parameter_supports(pref, unique_supports)
    end
    return
end

"""
    add_supports(pref::ParameterRef, supports::Union{Number, Vector{<:Number}})

Add additional support points for `pref`.

**Example**
```jldoctest; setup = :(using InfiniteOpt, JuMP; model = InfiniteModel(); @infinite_parameter(model, t in [0, 1], supports = [0, 1]))
julia> add_supports(t, 0.5)

julia> supports(t)
3-element Array{Float64,1}:
 0.0
 1.0
 0.5

julia> add_supports(t, [0.25, 1])

julia> supports(t)
4-element Array{Float64,1}:
 0.0
 1.0
 0.5
 0.25
```
"""
function add_supports(pref::ParameterRef, supports::Union{Number,
                                                          Vector{<:Number}})
    set = _parameter_set(pref)
    _check_supports_in_bounds(error, supports, set)
    current_supports = _parameter_supports(pref)
    if !(is_independent(pref)) &&
       sum(values(pref.model.param_to_group_id) .== group_id(pref)) > 1
        new_supports = [current_supports; supports]
    else
        new_supports = unique([current_supports; supports])
    end
    _update_parameter_supports(pref, new_supports)
    return
end

"""
    delete_supports(pref::ParameterRef)

Delete the support points for `pref`.

**Example**
```jldoctest; setup = :(using InfiniteOpt, JuMP; model = InfiniteModel(); @infinite_parameter(model, t in [0, 1], supports = [0, 1]))
julia> delete_supports(t)

julia> supports(t)
ERROR: Parameter t does not have supports.
```
"""
function delete_supports(pref::ParameterRef)
    _update_parameter_supports(pref, Int[])
    return
end

"""
    is_finite_parameter(pref::ParameterRef)::Bool

Return a `Bool` indicating if `pref` is a finite parameter.

**Example**
```jldoctest; setup = :(using InfiniteOpt, JuMP; model = InfiniteModel(); @finite_parameter(model, cost, 42))
julia> is_finite_parameter(cost)
true
```
"""
function is_finite_parameter(pref::ParameterRef)::Bool
    set = infinite_set(pref)
    if isa(set, IntervalSet) && set.lower_bound == set.upper_bound
        return true
    end
    return false
end

"""
    JuMP.value(pref::ParameterRef)::Number

Return the value of `pref` so long as it is a finite parameter. Errors if it is
an infinite parameter.

**Example**
```jldoctest; setup = :(using InfiniteOpt, JuMP; model = InfiniteModel(); @finite_parameter(model, cost, 42))
julia> value(cost)
42
```
"""
function JuMP.value(pref::ParameterRef)::Number
    !is_finite_parameter(pref) && error("$pref is an infinite parameter.")
    return supports(pref)[1]
end

"""
    JuMP.set_value(pref::ParameterRef, value::Number)

Set the value of `pref` so long as it is a finite parameter. Errors if it is
an infinite parameter.

**Example**
```jldoctest; setup = :(using InfiniteOpt, JuMP; model = InfiniteModel(); @finite_parameter(model, cost, 42))
julia> set_value(cost, 27)

julia> value(cost)
27
```
"""
function JuMP.set_value(pref::ParameterRef, value::Number)
    !is_finite_parameter(pref) && error("$pref is an infinite parameter.")
    set_infinite_set(pref, IntervalSet(value, value))
    set_supports(pref, [value], force = true)
    return
end

"""
    fill_in_supports!(model::InfiniteModel; [num_supports::Int = 10,
                      sig_fig::Int = 5])

Automatically generate support points for all infinite parameters in model. User
can specify the number of significant figures kept after decimal point for the
auto-generated supports wtih `sig_fig`. This calls
[`fill_in_supports!`](@ref fill_in_supports!(::ParameterRef)) for each parameter
in the model. See [`fill_in_supports!`](@ref fill_in_supports!(::ParameterRef))
for more information. Errors if one of the infinite set types is unrecognized.

**Example**
```jldoctest; setup = :(using InfiniteOpt; model = InfiniteModel(); @infinite_parameter(model, 0 <= x <= 1);)
julia> fill_in_supports!(model, num_supports = 4, sig_fig = 3)

julia> supports(x)
4-element Array{Number,1}:
 0.0
 0.333
 0.667
 1.0
```
"""
function fill_in_supports!(model::InfiniteModel; num_supports::Int = 10,
                           sig_fig::Int = 5)
    for key in keys(model.params)
        pref = ParameterRef(model, key)
        fill_in_supports!(pref, num_supports = num_supports, sig_fig = sig_fig)
    end
    return
end

"""
    fill_in_supports!(pref::ParameterRef; [num_supports::Int = 10,
                                           sig_fig::Int = 5])

Automatically generate support points for a particular infinite parameter `pref`.
Generating `num_supports` for the parameter. The supports are generated uniformly
if the underlying infinite set is an `IntervalSet` or they are generating randomly
accordingly to the distribution if the set is a `DistributionSet`.
User can specify the number of digits kept after decimal point for the
auto-generated supports wtih `sig_fig`. Extensions that use user defined
set types should extend [`generate_and_add_supports!`](@ref) and/or
[`generate_support_values`](@ref) as needed. Errors if the infinite set type is
not recognized.

**Example**
```jldoctest; setup = :(using InfiniteOpt; model = InfiniteModel(); @infinite_parameter(model, 0 <= x <= 1);)
julia> fill_in_supports!(x, num_supports = 4, sig_fig = 3)

julia> supports(x)
4-element Array{Number,1}:
 0.0
 0.333
 0.667
 1.0

```
"""
function fill_in_supports!(pref::ParameterRef; num_supports::Int = 10,
                           sig_fig::Int = 5)
    p = JuMP.owner_model(pref).params[JuMP.index(pref)]
    if length(p.supports) == 0
        generate_and_add_supports!(pref, p.set, num_supports = num_supports,
                                   sig_fig = sig_fig)
    end
    return
end

"""
    generate_and_add_supports!(pref::ParameterRef, set::AbstractInfiniteSet;
                               [num_supports::Int = 10, sig_fig::Int = 5])

Generate supports for `pref` via [`generate_support_values`](@ref) and add them
to `pref`. This is intended as an extendable internal method for
[`fill_in_supports!`](@ref fill_in_supports!(::ParameterRef)). Note that if
`pref` is part of a `DistributionSet` that features a multivariate distribution,
all the associated parameters with `pref` will also have supports added to them.
Most extensions that empoy user-defined infinite sets can typically enable this
by extending [`generate_support_values`](@ref). However, in some cases it may be
necessary to extend this when more complex operations need to take place then just
adding supports to a single infinite parameter (e.g., how we enable multivariate
distribution sets). Errors if the infinite set type is not recognized.
"""
function generate_and_add_supports!(pref::ParameterRef, set::AbstractInfiniteSet;
                                    num_supports::Int = 10, sig_fig::Int = 5)
    add_supports(pref, generate_support_values(set, num_supports = num_supports,
                                               sig_fig = sig_fig))
    return
end

# Multivariate distribution sets
function generate_and_add_supports!(pref::ParameterRef,
                                    set::DistributionSet{<:Distributions.MultivariateDistribution};
                                    num_supports::Int = 10, sig_fig::Int = 5)
    pref_group_id = group_id(pref)
    model = JuMP.owner_model(pref)
    associated_p_index = sort([i for i in 1:length(model.params)
                               if model.param_to_group_id[i] == pref_group_id])
    new_supports = generate_support_values(set, num_supports = num_supports, sig_fig = sig_fig)

    for i in 1:length(associated_p_index)
        pref_i = ParameterRef(model, associated_p_index[i])
        add_supports(pref_i, new_supports[i, :])
    end
    return
end

"""
    group_id(pref::ParameterRef)::Int

Return the group ID number for `pref`.

**Example**
```jldoctest; setup = :(using InfiniteOpt, JuMP; model = InfiniteModel(); @infinite_parameter(model, t in [0, 1], supports = [0, 1]))
julia> group_id(t)
1
```
"""
function group_id(pref::ParameterRef)::Int
    return JuMP.owner_model(pref).param_to_group_id[JuMP.index(pref)]
end

"""
    group_id(prefs::AbstractArray{<:ParameterRef})::Int

Return the group ID number for a group of `prefs`. Error if contains multiple
groups.

**Example**
```julia-repl
julia> group_id([x[1], x[2]])
2
```
"""
function group_id(prefs::AbstractArray{<:ParameterRef})::Int
    groups = group_id.(prefs)
    length(unique(groups)) != 1 && error("Array contains parameters from " *
                                         "multiple groups.")
    return first(groups)
end

"""
    is_independent(pref::ParameterRef)::Bool

Returns true for `pref` if it is independent or false otherwise.

**Example**
```jldoctest; setup = :(using InfiniteOpt, JuMP; model = InfiniteModel(); @infinite_parameter(model, t in [0, 1], supports = [0, 1]))
julia> is_independent(t)
false
```
"""
function is_independent(pref::ParameterRef)::Bool
    return JuMP.owner_model(pref).params[JuMP.index(pref)].independent
end

"""
    set_independent(pref::ParameterRef)

Specify that `pref` be independent.

**Example**
```jldoctest; setup = :(using InfiniteOpt, JuMP; model = InfiniteModel(); @infinite_parameter(model, t in [0, 1], supports = [0, 1]))
julia> set_independent(t)

julia> is_independent(t)
true
```
"""
function set_independent(pref::ParameterRef)
    old_param = JuMP.owner_model(pref).params[JuMP.index(pref)]
    new_param = InfOptParameter(old_param.set, old_param.supports, true)
    JuMP.owner_model(pref).params[JuMP.index(pref)] = new_param
    return
end

"""
    unset_independent(pref::ParameterRef)

Specify that `pref` be not independent.

**Example**
```jldoctest; setup = :(using InfiniteOpt, JuMP; model = InfiniteModel(); @infinite_parameter(model, t in [0, 1], supports = [0, 1]))
julia> unset_independent(t)

julia> is_independent(t)
false
```
"""
function unset_independent(pref::ParameterRef)
    old_param = JuMP.owner_model(pref).params[JuMP.index(pref)]
    new_param = InfOptParameter(old_param.set, old_param.supports, false)
    JuMP.owner_model(pref).params[JuMP.index(pref)] = new_param
    return
end

"""
    parameter_by_name(model::InfiniteModel, name::String)::Union{ParameterRef,
                                                                 Nothing}

Return the parameter reference assoociated with a parameter name. Errors if
multiple parameters have the same name. Returns nothing if no such name exists.

**Example**
```jldoctest; setup = :(using InfiniteOpt, JuMP; model = InfiniteModel(); @infinite_parameter(model, t in [0, 1], supports = [0, 1]))
julia> parameter_by_name(model, "t")
t
```
"""
function parameter_by_name(model::InfiniteModel,
                           name::String)::Union{ParameterRef, Nothing}
    if model.name_to_param === nothing
        # Inspired from MOI/src/Utilities/model.jl
        model.name_to_param = Dict{String, Int}()
        for (param, param_name) in model.param_to_name
            if haskey(model.name_to_param, param_name)
                # -1 is a special value that means this string does not map to
                # a unique variable name.
                model.name_to_param[param_name] = -1
            else
                model.name_to_param[param_name] = param
            end
        end
    end
    index = get(model.name_to_param, name, nothing)
    if index isa Nothing
        return
    elseif index == -1
        error("Multiple parameters have the name $name.")
    else
        return ParameterRef(model, index)
    end
end

"""
    all_parameters(model::InfiniteModel)::Vector{ParameterRef}

Return all of the infinite parameter references currently in `model`.

**Example**
```julia-repl
julia> all_parameters(model)
3-element Array{ParameterRef,1}:
 t
 x[1]
 x[2]
```
"""
function all_parameters(model::InfiniteModel)::Vector{ParameterRef}
    pref_list = Vector{ParameterRef}(undef, num_parameters(model))
    indexes = sort([index for index in keys(model.params)])
    counter = 1
    for index in indexes
        pref_list[counter] = ParameterRef(model, index)
        counter += 1
    end
    return pref_list
end

## Define functions to extract the names of parameters
# Extract the root name of a parameter reference
function _root_name(pref::ParameterRef)::String
    name = JuMP.name(pref)
    first_bracket = findfirst(isequal('['), name)
    if first_bracket == nothing
        return name
    else
        # Hacky fix to handle invalid Unicode
        try
            return name[1:first_bracket-1]
        catch
            return name[1:first_bracket-2]
        end
    end
end

# Return the root names of a tuple parameter of references
function _root_names(prefs::Tuple)::Tuple
    return _root_name.(first.(prefs))
end

## Internal functions for group checking
# Return group id of ParameterRef
function _group(pref::ParameterRef)::Int
    return group_id(pref)
end

# Return the group if of the first element in an array (assuming all same)
function _group(arr::AbstractArray{<:ParameterRef})::Int
    return group_id(first(arr))
end

# Return true if SparseAxisArray only has one group
function _only_one_group(arr::JuMPC.SparseAxisArray{<:ParameterRef})::Bool
    return length(unique(group_id.(arr))) == 1
end

# Return true to have one group ID since is singular
_only_one_group(pref::ParameterRef)::Bool = true

## Internal function for extracting parameter references
# Return a vector of parameter references from a tuple of references
function _list_parameter_refs(prefs::Tuple)
    list = ParameterRef[]
    for pref in prefs
        if isa(pref, ParameterRef)
            push!(list, pref)
        else
            for k in keys(pref.data)
                push!(list, pref[k])
            end
        end
    end
    return list
end
