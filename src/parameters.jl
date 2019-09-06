# Extend Base.copy for infinite parameters/variables etc.

function Base.copy(v::InfOptVariableRef, new_model::InfiniteModel)::InfOptVariableRef
    return InfOptVariableRef(new_model, v.index, variable_type(v))
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
        check = :(isa($(info.set), AbstractInfiniteSet))
        return :($(check) ? $(info.set) : error("Set must be a subtype of AbstractInfiniteSet."))
    else
        _error("Must specify upper/lower bounds, a distribution, or a set")
    end
    return
end

# Check that supports don't violate the set bounds
function _check_supports_in_bounds(_error::Function,
                                   supports::Union{Number, Vector{<:Number}},
                                   set::AbstractInfiniteSet)
    min_support = minimum(supports)
    max_support = maximum(supports)
    if isa(set, IntervalSet)
       if min_support < set.lower_bound || max_support > set.upper_bound
           _error("Support points violate the interval set bounds.")
       end
   elseif isa(set, DistributionSet{<:Distributions.UnivariateDistribution})
       check1 = min_support < minimum(set.distribution)
       check2 = max_support > maximum(set.distribution)
       if check1 || check2
           _error("Support points violate the distribution set bounds.")
       end
    end
    return
end

"""
    build_parameter(_error::Function, set::AbstractInfiniteSet,
                    [num_params::Int64 = 1;
                    supports::Union{Number, Vector{<:Number}} = Number[],
                    independent::Bool = false])::InfOptParameter

Returns a [`InfOptParameter`](@ref) given the appropriate information. This is
analagous to `JuMP.build_variable`. Errors if supports violate the bounds
associated `set`. Also errors if `set` contains a multivariate distribution with
a different dimension than `num_params`. This is meant to primarily serve as a
helper method for [`@infinite_parameter`](@ref).

**Example**
```julia
julia> build_parameter(error, IntervalSet(0, 3), supports = Vector(0:3))
InfOptParameter{IntervalSet}(IntervalSet(0.0, 3.0), [0, 1, 2, 3], false)
```
"""
function build_parameter(_error::Function, set::AbstractInfiniteSet,
                         num_params::Int64 = 1;
                         supports::Union{Number, Vector{<:Number}} = Number[],
                         independent::Bool = false,
                         extra_kw_args...)::InfOptParameter
    for (kwarg, _) in extra_kw_args
        _error("Unrecognized keyword argument $kwarg")
    end
    if length(supports) != 0
        _check_supports_in_bounds(_error, supports, set)
    end
    if isa(set, DistributionSet{<:Distributions.MultivariateDistribution})
        if num_params != length(set.distribution)
            _error("Multivariate distribution dimension must match dimension " *
                   "of parameter.")
        end
    end
    unique_supports = sort(unique(supports))
    if length(unique_supports) != length(supports)
        @warn("Support points are not unique, eliminating redundant points.")
    end
    return InfOptParameter(set, unique_supports, independent)
end

"""
    add_parameter(model::InfiniteModel, p::InfOptParameter,
                  name::String="")::InfOptVariableRef

Returns a [`InfOptVariableRef`](@ref) associated with the parameter `p` that is
added to `model`. This adds a parameter to the model in a manner similar to
`JuMP.add_variable`. This can be used to add parameters with the use of
[`@infinite_parameter`](@ref). [`build_parameter`](@ref) should be used to
construct `p`.

**Example**
```julia
julia> p = build_parameter(error, IntervalSet(0, 3), supports = Vector(0:3))
InfOptParameter{IntervalSet}(IntervalSet(0.0, 3.0), [0, 1, 2, 3], false)

julia> param_ref = add_parameter(model, p, "name")
name
```
"""
function add_parameter(model::InfiniteModel, p::InfOptParameter,
                       name::String=""; macro_call = false)::InfOptVariableRef
    model.next_param_index += 1
    pref = InfOptVariableRef(model, model.next_param_index, Parameter)
    model.params[JuMP.index(pref)] = p
    if !macro_call
        model.next_param_id += 1
    end
    model.param_to_group_id[JuMP.index(pref)] = model.next_param_id
    JuMP.set_name(pref, name)
    return pref
end

"""
    used_by_constraint(ref::InfOptVariableRef)::Bool

Return true if `ref` is used by a constraint or false otherwise.

**Example**
```julia
julia> used_by_constraint(t)
true
```
"""
# Function wrapper of used_by_constraint
function used_by_constraint(ref::InfOptVariableRef)::Bool
    return used_by_constraint(ref, Val(variable_type(ref)))
end

# used_by_constraint for parameters
function used_by_constraint(pref::InfOptVariableRef, ::Val{Parameter})::Bool
    return haskey(JuMP.owner_model(pref).param_to_constrs, JuMP.index(pref))
end

"""
    used_by_measure(ref::InfOptVariableRef)::Bool

Return true if `ref` is used by a measure or false otherwise.

**Example**
```julia
julia> used_by_measure(t)
false
```
"""
function used_by_measure(ref::InfOptVariableRef)::Bool
    return used_by_measure(ref, Val(variable_type(ref)))
end

function used_by_measure(pref::InfOptVariableRef, ::Val{Parameter})::Bool
    return haskey(JuMP.owner_model(pref).param_to_meas, JuMP.index(pref))
end

"""
    used_by_variable(pref::InfOptVariableRef)::Bool

Return true if `pref` is used by an infinite variable or false otherwise.

**Example**
```julia
julia> used_by_variable(t)
true
```
"""
function used_by_variable(pref::InfOptVariableRef)::Bool
    type = variable_type(pref)
    if type != Parameter
        error("pref must be a parameter reference.")
    end
    return haskey(JuMP.owner_model(pref).param_to_vars, JuMP.index(pref))
end

"""
    is_used(pref::InfOptVariableRef)::Bool

Return true if `pref` is used in the model or false otherwise.

**Example**
```julia
julia> is_used(t)
true
```
"""
# Function wrapper for is_used
function is_used(pref::InfOptVariableRef)::Bool
    return is_used(pref, ::Val(variable_type(pref)))
end

# is_used for parameters
function is_used(pref::InfOptVariableRef, ::Val{Parameter})::Bool
    return used_by_measure(pref) || used_by_constraint(pref) || used_by_variable(pref)
end

## Check if parameter is used by measure data and error if it is to prevent bad
## deleting behavior
# DiscreteMeasureData
function _check_param_in_data(pref::InfOptVariableRef, data::DiscreteMeasureData)
    type = variable_type(pref)
    if type != Parameter
        error("pref must be a parameter reference.")
    end
    data.parameter_ref == pref && error("Unable to delete $pref since it is " *
                                        "used to evaluate measures.")
    return
end

# MultiDiscreteMeasureData
function _check_param_in_data(pref::InfOptVariableRef, data::MultiDiscreteMeasureData)
    type = variable_type(pref)
    if type != Parameter
        error("pref must be a parameter reference.")
    end
    pref in data.parameter_ref && error("Unable to delete $pref since it is " *
                                        "used to evaluate measures.")
    return
end

# Fallback
function _check_param_in_data(pref::InfOptVariableRef, data::T) where {T}
    type = variable_type(pref)
    if type != Parameter
        error("pref must be a parameter reference.")
    end
    error("Unable to delete parameters when using measure data type $T.")
    return
end

## Determine if a tuple element contains a particular parameter
# InfOptVariableRef
function _contains_pref(search_pref::InfOptVariableRef, pref::InfOptVariableRef)::Bool
    if variable_type(search_pref) != Parameter
        error("search_pref must be a parameter reference.")
    end
    if variable_type(pref) != Parameter
        error("pref must be a parameter reference.")
    end
    return search_pref == pref
end

# SparseAxisArray
function _contains_pref(arr::JuMPC.SparseAxisArray{<:InfOptVariableRef},
                        pref::InfOptVariableRef)::Bool
    if variable_type(pref) != Parameter
        error("pref must be a parameter reference.")
    end
    return pref in collect(values(arr.data))
end

# Return parameter tuple without a particular parameter and return location of
# where it was
function _remove_parameter(prefs::Tuple, delete_pref::InfOptVariableRef)::Tuple
    if variable_type(delete_pref) != Parameter
        error("delete_pref must be a parameter reference.")
    end
    for i = 1:length(prefs)
        if _contains_pref(prefs[i], delete_pref)
            if variable_type(prefs[i]) == Parameter
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
function _update_infinite_variable(vref::InfOptVariableRef, new_prefs::Tuple)
    _update_infinite_variable(vref, Val(variable_type(vref)), new_prefs)
    return
end

function _update_infinite_variable(vref::InfOptVariableRef,
                                   ::Val{Infinite}, new_prefs::Tuple)
    _update_variable_param_refs(vref, new_prefs)
    JuMP.set_name(vref, _root_name(vref))
    if used_by_measure(vref)
        for mindex in JuMP.owner_model(vref).var_to_meas[JuMP.index(vref)]
            JuMP.set_name(InfOptVariableRef(JuMP.owner_model(vref), mindex, Val(MeasureRef)),
                       _make_meas_name(JuMP.owner_model(vref).measures[mindex]))
        end
    end
    return
end

# Return a parameter value tuple without element at a particular location
function _remove_parameter_values(pref_vals::Tuple, location::Tuple)::Tuple
    # removed parameter was a scalar parameter
    if length(location) == 1
        return Tuple(pref_vals[i] for i = 1:length(pref_vals) if i != location[1])
    # removed parameter was part of an array
    else
        new_dict = filter(x -> x.first != location[2], pref_vals[location[1]].data)
        val_list = [pref_vals...]
        val_list[location[1]] = JuMPC.SparseAxisArray(new_dict)
        return Tuple(val_list[i] for i = 1:length(pref_vals))
    end
end

# Update point variable for which a parameter is deleted
function _update_point_variable(pvref::InfOptVariableRef, pref_vals::Tuple)
    _update_point_variable(pvref, Val(variable_type(pvref)), pref_vals)
    return
end

function _update_point_variable(pvref::InfOptVariableRef,
                                ::Val{Point}, pref_vals::Tuple)
    _update_variable_param_values(pvref, pref_vals)
    # update name if no alias was provided
    if !isa(findfirst(isequal('('), JuMP.name(pvref)), Nothing)
        JuMP.set_name(pvref, "")
    end
    if used_by_measure(pvref)
        for mindex in JuMP.owner_model(pvref).var_to_meas[JuMP.index(pvref)]
            JuMP.set_name(InfOptVariableRef(JuMP.owner_model(pvref), mindex, Val(MeasureRef)),
                      _make_meas_name(JuMP.owner_model(pvref).measures[mindex]))
        end
    end
    return
end

# Update a reduced variable associated with an infinite variable whose parameter
# was removed
function _update_reduced_variable(vref::InfOptVariableRef,
                                  location::Tuple)
    if variable_type(vref) != Reduced
        error("vref must be a reduced infinite variable reference.")
    end
    eval_supps = eval_supports(vref)
    # removed parameter was a scalar
    if length(location) == 1
        new_supports = Dict{Int64, Union{Number, JuMPC.SparseAxisArray}}()
        for (index, support) in eval_supps
            if index < location[1]
                new_supports[index] = support
            elseif index > location[1]
                new_supports[index - 1] = support
            end
        end
        JuMP.owner_model(vref).reduced_info[JuMP.index(vref)] = ReducedInfiniteInfo(infinite_variable_ref(vref), new_supports)
    # removed parameter was part of an array and was reduced previously
    elseif haskey(eval_supps, location[1])
        new_dict = filter(x -> x.first != location[2], eval_supps[location[1]].data)
        new_supports = copy(eval_supps)
        new_supports[location[1]] = JuMPC.SparseAxisArray(new_dict)
        JuMP.owner_model(vref).reduced_info[JuMP.index(vref)] = ReducedInfiniteInfo(infinite_variable_ref(vref), new_supports)
    end
    if used_by_measure(vref)
        for mindex in JuMP.owner_model(vref).reduced_to_meas[JuMP.index(vref)]
            JuMP.set_name(InfOptVariableRef(JuMP.owner_model(vref), mindex, Val(MeasureRef)),
                       _make_meas_name(JuMP.owner_model(vref).measures[mindex]))
        end
    end
    return
end

# Function wrapper for JuMP.delete extension
function JuMP.delete(model::InfiniteModel, ref::InfOptVariableRef)
    return JuMP.delete(model, ref, Val(variable_type(ref)))
end

"""
    JuMP.delete(model::InfiniteModel, pref::InfOptVariableRef, ::Val{Parameter})

Extend [`JuMP.delete`](@ref) to delete infinite parameters and their
dependencies. All variables, constraints, and measure functions that depend on
`pref` are updated to exclude it. Errors if the parameter is contained in an
`AbstractMeasureData` datatype that is employed by a measure since the measure
becomes invalid otherwise. Thus, measures that contain this dependency must
be deleted first. Note that `_check_param_in_data(pref, measure_data)` needs to
be extended to allow deletion of parameters when custom `AbstractMeasureData`
datatypes are used.

**Example**
```julia
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
# JuMP.delete for parameters
function JuMP.delete(model::InfiniteModel, pref::InfOptVariableRef, ::Val{Parameter})
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
            if variable_type(model.measures[mindex].func) == Parameter
                model.measures[mindex] = Measure(zero(JuMP.AffExpr),
                                                 model.measures[mindex].data)
            else
                _remove_variable(model.measures[mindex].func, pref)
            end
            JuMP.set_name(InfOptVariableRef(model, mindex, Val(MeasureRef)),
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
            vref = InfOptVariableRef(model, vindex, Infinite)
            # remove the parameter dependence
            _update_infinite_variable(vref, prefs)
            # update any point variables that depend on vref accordingly
            if used_by_point_variable(vref)
                for pindex in model.infinite_to_points[vindex]
                    pvref = InfOptVariableRef(model, pindex, Point)
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
            if variable_type(model.constrs[cindex].func) == Parameter
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
JuMP.is_valid(model::InfiniteModel, pref::InfOptVariableRef)::Bool

Extend the [`JuMP.is_valid`](@ref) function to accomodate infinite parameters.
Returns true if the `InfiniteModel` stored in `ref` matches `model` and if
    the index is used by `model`. It returns false otherwise.

    **Example**
    ```julia
    julia> is_valid(model, t)
    true
    ```
"""

# Function wrapper for JuMP.is_valid extension
function JuMP.is_valid(model::InfiniteModel, ref::InfOptVariableRef)::Bool
    return JuMP.is_valid(model::InfiniteModel,
                         ref::InfOptVariableRef,
                         ::variable_type(ref))
end

# JuMP.is_valid for parameters
function JuMP.is_valid(model::InfiniteModel, pref::InfOptVariableRef,
                       ::Val{Parameter})::Bool
    check1 = model === JuMP.owner_model(pref)
    check2 = JuMP.index(pref) in keys(model.params)
    return check1 && check2
end

"""
    JuMP.name(ref::InfOptVariableRef)::String

Extend the `JuMP.name` function to accomodate InfOptVariableRef.
Returns the name string associated with `ref`.

**Example**
```julia
julia> name(t)
"t"
```
"""
# function wrapper for all JuMP.name functions
function JuMP.name(ref::InfOptVariableRef)::String
    return JuMP.name(ref, Val(variable_type(ref)))
end

# JuMP.name for parameter
function JuMP.name(pref::InfOptVariableRef, ::Val{Parameter})::String
    return JuMP.owner_model(pref).param_to_name[JuMP.index(pref)]
end

"""
    JuMP.set_name(ref::InfOptVariableRef, name::String)

Extend the `JuMP.set_name` function to accomodate `InfOptVariableRef`. Set a new
base name to be associated with `ref`.

**Example**
```julia
julia> set_name(t, "time")

julia> name(t)
"time"
```
"""

# Function wrapper for JuMP.set_name
function JuMP.set_name(ref::InfOptVariableRef, name::String)
    JuMP.set_name(ref, Val(variable_type(ref)), name)
end

# JuMP.set_name for parameter
function JuMP.set_name(pref::InfOptVariableRef, ::Val{Parameter}, name::String)
    JuMP.owner_model(pref).param_to_name[JuMP.index(pref)] = name
    JuMP.owner_model(pref).name_to_param = nothing
    return
end

"""
    num_parameters(model::InfiniteModel)::Int64

Return the number of infinite parameters currently present in `model`.

**Example**
```julia
julia> num_parameters(model)
1
```
"""
function num_parameters(model::InfiniteModel)::Int64
    return length(model.params)
end

# Internal functions
_parameter_set(pref::InfOptVariableRef) = JuMP.owner_model(pref).params[JuMP.index(pref)].set
_parameter_supports(pref::InfOptVariableRef) = JuMP.owner_model(pref).params[JuMP.index(pref)].supports
function _update_parameter_set(pref::InfOptVariableRef, set::AbstractInfiniteSet)
    supports = JuMP.owner_model(pref).params[JuMP.index(pref)].supports
    independent = JuMP.owner_model(pref).params[JuMP.index(pref)].independent
    JuMP.owner_model(pref).params[JuMP.index(pref)] = InfOptParameter(set, supports, independent)
    if is_used(pref)
        set_optimizer_model_ready(JuMP.owner_model(pref), false)
    end
    return
end
function _update_parameter_supports(pref::InfOptVariableRef, supports::Vector{<:Number})
    set = JuMP.owner_model(pref).params[JuMP.index(pref)].set
    independent = JuMP.owner_model(pref).params[JuMP.index(pref)].independent
    JuMP.owner_model(pref).params[JuMP.index(pref)] = InfOptParameter(set, supports, independent)
    if is_used(pref)
        set_optimizer_model_ready(JuMP.owner_model(pref), false)
    end
    return
end

"""
    infinite_set(pref::InfOptVariableRef)::AbstractInfiniteSet

Return the infinite set associated with `pref`.

**Example**
```julia
julia> infinite_set(t)
IntervalSet(0.0, 3.0)
```
"""
function infinite_set(pref::InfOptVariableRef)::AbstractInfiniteSet
    return _parameter_set(pref)
end

"""
    set_infinite_set(pref::InfOptVariableRef, set::AbstractInfiniteSet)

Specify the infinite set of `pref`.

**Example**
```julia
julia> set_infinite_set(t, IntervalSet(0, 1))

julia> infinite_set(t)
IntervalSet(0.0, 1.0)
```
"""
function set_infinite_set(pref::InfOptVariableRef, set::AbstractInfiniteSet)
    _update_parameter_set(pref, set)
    return
end

"""
    JuMP.has_lower_bound(ref::InfOptVariableRef)::Bool

Extend the `JuMP.has_lower_bound` function to accomodate InfOptVariableRef.
Return true if the set associated with `ref` has a defined lower bound or if a
lower bound can be found.

**Example**
```julia
julia> has_lower_bound(t)
true
```
"""
# Function wrapper for JuMP.has_lower_bound
function JuMP.has_lower_bound(ref::InfOptVariableRef)::Bool
    return JuMP.has_lower_bound(ref, Val(variable_type(ref)))
end

# JuMP.has_lower_bound for parameters
function JuMP.has_lower_bound(pref::InfOptVariableRef, ::Val{Parameter})::Bool
    set = _parameter_set(pref)
    if isa(set, IntervalSet)
        return true
    elseif isa(set, DistributionSet)
        if typeof(set.distribution) <: Distributions.UnivariateDistribution
            return true
        else
            false
        end
    else
        type = typeof(set)
        error("Undefined infinite set type $type for lower bound checking.")
    end
end

"""
    JuMP.lower_bound(ref::InfOptVariableRef)::Number

Extend the `JuMP.lower_bound` function to accomodate InfOptVariableRef.
Returns the lower bound associated with the infinite set. Errors if such a bound
is not well-defined.

**Example**
```julia
julia> lower_bound(t)
0.0
```
"""

# Function wrapper for JuMP.lower_bound
function JuMP.lower_bound(ref::InfOptVariableRef)::Number
    return JuMP.lower_bound(ref, Val(variable_type(ref)))
end

# JuMP.lower_bound for parameters
function JuMP.lower_bound(pref::InfOptVariableRef, ::Val{Parameter})::Number
    set = _parameter_set(pref)
    if !JuMP.has_lower_bound(pref)
        error("Parameter $(pref) does not have a lower bound.")
    end
    if isa(set, IntervalSet)
        return set.lower_bound
    else
        return minimum(set.distribution)
    end
end

"""
    JuMP.set_lower_bound(ref::InfOptVariableRef, lower::Number)

Extend the `JuMP.set_lower_bound` function to accomodate infinite parameters and
variables. Updates the infinite set lower bound if and only if it is an
IntervalSet. Errors otherwise.

**Example**
```julia
julia> set_lower_bound(t, -1)

julia> lower_bound(t)
-1.0
```
"""
# Function wrapper for JuMP.set_lower_bound
function JuMP.set_lower_bound(ref::InfOptVariableRef, lower::Number)
    JuMP.set_lower_bound(ref, Val(variable_type(ref)), lower)
end

function JuMP.set_lower_bound(pref::InfOptVariableRef, ::Val{Parameter},
                              lower::Number)
    set = _parameter_set(pref)
    if isa(set, DistributionSet)
        error("Cannot set the lower bound of a distribution, try using " *
              "`Distributions.Truncated` instead.")
    elseif !isa(set, IntervalSet)
        error("Parameter $(pref) is not an interval set.")
    end
    _update_parameter_set(pref, IntervalSet(lower, set.upper_bound))
    return
end

"""
    JuMP.has_upper_bound(ref::InfOptVariableRef)::Bool

Extend the `JuMP.has_upper_bound` function to accomodate infinite parameters and
variables. Return true if the set associated with `ref` has a defined upper
bound or if an upper bound can be found.

**Example**
```julia
julia> has_upper_bound(t)
true
```
"""
# Function wrapper for JuMP.has_upper_bound
function JuMP.has_upper_bound(ref::InfOptVariableRef)::Bool
    return JuMP.has_upper_bound(ref, Val(variable_type(ref)))
end

# JuMP.has_upper_bound for parameters
function JuMP.has_upper_bound(pref::InfOptVariableRef, ::Val{Parameter})::Bool
    set = _parameter_set(pref)
    if isa(set, IntervalSet)
        return true
    elseif isa(set, DistributionSet)
        if typeof(set.distribution) <: Distributions.UnivariateDistribution
            return true
        else
            false
        end
    else
        type = typeof(set)
        error("Undefined infinite set type $type for lower upper checking.")
    end
end

"""
    JuMP.upper_bound(pref::InfOptVariableRef)::Number

Extend the `JuMP.upper_bound` function to accomodate infinite parameters and
variables. Returns the upper bound associated with the infinite set. Errors if
such a bound is not well-defined.

**Example**
```julia
julia> upper_bound(t)
1.0
```
"""
# Function wrapper for JuMP.upper_bound
function JuMP.upper_bound(ref::InfOptVariableRef)::Number
    return JuMP.upper_bound(ref, variable_type(ref))
end

# JuMP.upper_bound for parameters
function JuMP.upper_bound(pref::InfOptVariableRef, ::Val{Parameter})::Number
    set = _parameter_set(pref)
    if !JuMP.has_upper_bound(pref)
        error("Parameter $(pref) does not have a upper bound.")
    end
    if isa(set, IntervalSet)
        return set.upper_bound
    else
        return maximum(set.distribution)
    end
end

"""
    JuMP.set_upper_bound(pref::InfOptVariableRef, lower::Number)

Extend the `JuMP.set_upper_bound` function to accomodate infinite parameters and
variables. Updates the infinite set upper bound if and only if it is an
IntervalSet. Errors otherwise.

**Example**
```julia
julia> set_upper_bound(t, 2)

julia> upper_bound(t)
2.0
```
"""
# Function wrapper for JuMP.set_upper_bound
function JuMP.set_upper_bound(ref::InfOptVariableRef, upper::Number)
    JuMP.set_upper_bound(ref, Val(variable_type(ref)), upper)
end

# JuMP.set_upper_bound for parameter
function JuMP.set_upper_bound(pref::InfOptVariableRef, ::Val{Parameter}, upper::Number)
    set = _parameter_set(pref)
    if isa(set, DistributionSet)
        error("Cannot set the upper bound of a distribution, try using " *
              "`Distributions.Truncated` instead.")
    elseif !isa(set, IntervalSet)
        error("Parameter $(pref) is not an interval set.")
    end
    _update_parameter_set(pref, IntervalSet(set.lower_bound, upper))
    return
end

"""
    num_supports(pref::InfOptVariableRef)::Int64

Return the number of support points associated with `pref`.

**Example**
```julia
julia> num_supports(t)
1
```
"""
function num_supports(pref::InfOptVariableRef)::Int64
    return length(_parameter_supports(pref))
end

"""
    has_supports(pref::InfOptVariableRef)::Bool

Return true if `pref` has supports or false otherwise.

**Example**
```julia
julia> has_supports(t)
true
```
"""
has_supports(pref::InfOptVariableRef)::Bool = num_supports(pref) > 0

"""
    supports(pref::InfOptVariableRef)::Vector

Return the support points associated with `pref`. Errors if there are no
supports.

**Example**
```julia
julia> supports(t)
1-element Array{Int64,1}:
 1
```
"""
function supports(pref::InfOptVariableRef)::Vector
    !has_supports(pref) && error("Parameter $pref does not have supports.")
    return _parameter_supports(pref)
end

"""
    supports(prefs::AbstractArray{<:InfOptVariableRef})::Vector

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
```julia
julia> x = @infinite_parameter(model, [i = 1:2], set = IntervalSet(-1, 1),
                               base_name = "x", independent = true)
2-element Array{InfOptVariableRef,1}:
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
function supports(prefs::AbstractArray{<:InfOptVariableRef})::Vector
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
        for i = 1:length(support_list)
            support_list[i] = JuMPC.SparseAxisArray(Dict(k => supports(pref)[i] for (k, pref) in prefs.data))
        end
    else
        all_keys = collect(keys(prefs))
        all_supports = [supports(pref) for (k, pref) in prefs.data]
        support_list = Vector{JuMPC.SparseAxisArray}(undef,
                                                               prod(lengths))
        counter = 1
        for combo in Iterators.product(all_supports...)
            support_list[counter] = JuMPC.SparseAxisArray(Dict(all_keys[i] => combo[i] for i = 1:length(combo)))
            counter += 1
        end
    end
    return support_list
end

"""
    set_supports(pref::InfOptVariableRef, supports::Vector{<:Number})

Specify the support points for `pref`. Errors if the supports violate the bounds
associated with the infinite set. Warns if the points are not unique. Note that
this will overwrite existing supports.

**Example**
```julia
julia> set_supports(t, [0, 1])

julia> supports(t)
2-element Array{Int64,1}:
 0
 1
```
"""
function set_supports(pref::InfOptVariableRef, supports::Vector{<:Number})
    set = _parameter_set(pref)
    _check_supports_in_bounds(error, supports, set)
    unique_supports = sort(unique(supports))
    if length(unique_supports) != length(supports)
        @warn("Support points are not unique, eliminating redundant points.")
    end
    _update_parameter_supports(pref, unique_supports)
    return
end

"""
    add_supports(pref::InfOptVariableRef, supports::Union{Number, Vector{<:Number}})

Add additional support points for `pref`.

**Example**
```julia
julia> add_supports(t, 0.5)

julia> supports(t)
3-element Array{Float64,1}:
 0.0
 0.5
 1.0

julia> add_supports(t, [0.25, 1])

julia> supports(t)
4-element Array{Float64,1}:
 0.0
 0.25
 0.5
 1.0
```
"""
function add_supports(pref::InfOptVariableRef, supports::Union{Number,
                                                          Vector{<:Number}})
    set = _parameter_set(pref)
    _check_supports_in_bounds(error, supports, set)
    current_supports = _parameter_supports(pref)
    _update_parameter_supports(pref, sort(unique([current_supports; supports])))
    return
end

"""
    delete_supports(pref::InfOptVariableRef)

Delete the support points for `pref`.

**Example**
```julia
julia> delete_supports(t)

julia> supports(t)
ERROR: Parameter test does not have supports.
```
"""
function delete_supports(pref::InfOptVariableRef)
    _update_parameter_supports(pref, Number[])
    return
end

"""
    group_id(pref::InfOptVariableRef)::Int64

Return the group ID number for `pref`.

**Example**
```julia
julia> group_id(t)
1
```
"""
function group_id(pref::InfOptVariableRef)::Int64
    return JuMP.owner_model(pref).param_to_group_id[JuMP.index(pref)]
end

"""
    group_id(prefs::AbstractArray{<:InfOptVariableRef})::Int64

Return the group ID number for a group of `prefs`. Error if contains multiple
groups.

**Example**
```julia
julia> group_id([x[1], x[2]])
2
```
"""
function group_id(prefs::AbstractArray{<:InfOptVariableRef})::Int64
    groups = group_id.(prefs)
    length(unique(groups)) != 1 && error("Array contains parameters from " *
                                         "multiple groups.")
    return first(groups)
end

"""
    is_independent(pref::InfOptVariableRef)::Bool

Returns true for `pref` if it is independent or false otherwise.

**Example**
```julia
julia> is_independent(t)
false
```
"""
function is_independent(pref::InfOptVariableRef)::Bool
    return JuMP.owner_model(pref).params[JuMP.index(pref)].independent
end

"""
    set_independent(pref::InfOptVariableRef)

Specify that `pref` be independent.

**Example**
```julia
julia> set_independent(t)

julia> is_independent(t)
true
```
"""
function set_independent(pref::InfOptVariableRef)
    old_param = JuMP.owner_model(pref).params[JuMP.index(pref)]
    new_param = InfOptParameter(old_param.set, old_param.supports, true)
    JuMP.owner_model(pref).params[JuMP.index(pref)] = new_param
    return
end

"""
    unset_independent(pref::InfOptVariableRef)

Specify that `pref` be not independent.

**Example**
```julia
julia> unset_independent(t)

julia> is_independent(t)
false
```
"""
function unset_independent(pref::InfOptVariableRef)
    old_param = JuMP.owner_model(pref).params[JuMP.index(pref)]
    new_param = InfOptParameter(old_param.set, old_param.supports, false)
    JuMP.owner_model(pref).params[JuMP.index(pref)] = new_param
    return
end

"""
    parameter_by_name(model::InfiniteModel, name::String)::Union{InfOptVariableRef,
                                                                 Nothing}

Return the parameter reference assoociated with a parameter name. Errors if
multiple parameters have the same name. Returns nothing if no such name exists.

**Example**
```julia
julia> parameter_by_name(model, "t")
t
```
"""
function parameter_by_name(model::InfiniteModel,
                           name::String)::Union{InfOptVariableRef, Nothing}
    if model.name_to_param === nothing
        # Inspired from MOI/src/Utilities/model.jl
        model.name_to_param = Dict{String, Int64}()
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
        return InfOptVariableRef(model, index, Parameter)
    end
    return
end

"""
    all_parameters(model::InfiniteModel)::Vector{InfOptVariableRef}

Return all of the infinite parameter references currently in `model`.

**Example**
```julia
julia> all_parameters(model)
3-element Array{InfOptVariableRef,1}:
 t
 x[1]
 x[2]
```
"""
function all_parameters(model::InfiniteModel)::Vector{InfOptVariableRef}
    pref_list = Vector{InfOptVariableRef}(undef, num_parameters(model))
    indexes = sort([index for index in keys(model.params)])
    counter = 1
    for index in indexes
        pref_list[counter] = InfOptVariableRef(model, index, Parameter)
        counter += 1
    end
    return pref_list
end

## Define functions to extract the names of parameters
# Extract the root name of a parameter reference
function _root_name(pref::InfOptVariableRef)::String
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
# Return group id of parameter
function _group(pref::InfOptVariableRef)::Int64
    return group_id(pref)
end

# Return the group if of the first element in an array (assuming all same)
function _group(arr::AbstractArray{<:InfOptVariableRef})::Int64
    return group_id(first(arr))
end

# Return true if SparseAxisArray only has one group
function _only_one_group(arr::JuMPC.SparseAxisArray{<:InfOptVariableRef})::Bool
    return length(unique(group_id.(arr))) == 1
end

# Return true to have one group ID since is singular
_only_one_group(pref::InfOptVariableRef)::Bool = true

## Internal function for extracting parameter references
# Return a vector of parameter references from a tuple of references
function _list_parameter_refs(prefs::Tuple)
    list = InfOptVariableRef[]
    for pref in prefs
        if variable_type(pref) == Parameter
            push!(list, pref)
        else
            for k in keys(pref.data)
                push!(list, pref[k])
            end
        end
    end
    return list
end
