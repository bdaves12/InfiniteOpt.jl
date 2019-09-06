# Extend other Base functions
function Base.:(==)(v::T, w::U)::Bool where {T :: InfOptVariableRef,
                                             U :: InfOptVariableRef}
    return v.model === w.model && v.index == w.index
        && variable_type(v) == variable_type(w)
end

Base.broadcastable(v::InfOptVariableRef) = Ref(v)

# Extend JuMP functions
JuMP.isequal_canonical(v::InfOptVariableRef, w::InfOptVariableRef) = v == w
JuMP.variable_type(model::InfiniteModel) = InfOptVariableRef

function JuMP.variable_type(model::InfiniteModel, type::Symbol)
    if type in [Infinite, Point, Global, Parameter]
        return InfOptVariableRef
    else
        error("Invalid variable type.")
    end
end

# Check parameter tuple, ensure all elements contain parameter references
function _check_parameter_tuple(_error::Function, prefs::Tuple)
    types = [variable_type(pref) for pref in prefs]
    num_params = length(types)
    valid_types = zeros(Bool, num_params)
    for i in eachindex(types)
        if types[i] == Parameter
            valid_types[i] = true
        end
    end
    if !all(valid_types)
        _error("Invalid parameter type(s) given.")
    end
    return
end

# Convert parameter tuple s.t. array elements are SparseAxisArrays
function _make_formatted_tuple(prefs::Tuple)::Tuple
    converted_prefs = ()
    for pref in prefs
#        if isa(pref, ParameterRef) || isa(pref, Number)
        if variable_type(pref) == Parameter || isa(pref, Number)
            converted_prefs = (converted_prefs..., pref)
        else
            converted_prefs = (converted_prefs...,
                               convert(JuMPC.SparseAxisArray, pref))
        end
    end
    return converted_prefs
end

# Ensure each element onyl contains parameters with same group ID
function _check_tuple_groups(_error::Function, prefs::Tuple)
    valid_elements = _only_one_group.(prefs)
    if sum(valid_elements) != length(prefs)
        _error("Each parameter tuple element must have contain only infinite " *
               "parameters with the same group ID.")
    end
    groups = _group.(prefs)
    if length(unique(groups)) != length(groups)
        _error("Cannot double specify infinite parameter references.")
    end
    return
end

# Ensure parameter values match shape of parameter reference tuple stored in the
# infinite variable reference
function _check_tuple_shape(_error::Function,
                            infinite_variable_ref::InfOptVariableRef,
                            values::Tuple)
    _check_tuple_shape(_error, infinite_variable_ref,
                       Val(variable_type(infinite_variable_ref)), values)
    return
end

function _check_tuple_shape(_error::Function,
                            infinite_variable_ref::InfOptVariableRef,
                            ::Val{Infinite},
                            values::Tuple)
    prefs = parameter_refs(infinite_variable_ref)
    container = JuMPC.SparseAxisArray
    if length(prefs) != length(values)
        _error("The dimensions of the infinite parameter values must match " *
               "those defined for the infinite variable.")
    end
    for i in eachindex(values)
        if isa(prefs[i], InfOptVariableRef) && variable_type(prefs[i] == Parameter)
                                            && !(isa(values[i], Number))
            _error("The dimensions and array type of the infinite parameter " *
                   "values must match those defined for the infinite variable.")
        elseif isa(prefs[i], container) && !isa(values[i], container)
            _error("The dimensions and array type of the infinite parameter " *
                   "values must match those defined for the infinite variable.")
        elseif isa(prefs[i], container)
            if keys(prefs[i].data) != keys(values[i].data)
                _error("Index keys of infinite parameter values don't match " *
                       "those defined for the infinite variable.")
            end
        end
    end
    return
end

# Used to ensure values don't violate parameter bounds
function _check_tuple_values(_error::Function,
                             inf_vref::InfOptVariableRef,
                             param_values::Tuple)
    _check_tuple_values(_error, inf_vref, Val(variable_type(inf_vref)), param_values)
    return
end
function _check_tuple_values(_error::Function,
                             inf_vref::InfOptVariableRef, ::Val{Infinite},
                             param_values::Tuple)
    prefs = parameter_refs(inf_vref)
    for i in eachindex(prefs)
        if variable_type(prefs[i]) == Parameter
            if JuMP.has_lower_bound(prefs[i])
                check1 = param_values[i] < JuMP.lower_bound(prefs[i])
                check2 = param_values[i] > JuMP.upper_bound(prefs[i])
                if check1 || check2
                    _error("Parameter values violate parameter bounds.")
                end
            end
        else
            for (k, v) in prefs[i].data
                if JuMP.has_lower_bound(v)
                    check1 = param_values[i].data[k] < JuMP.lower_bound(v)
                    check2 = param_values[i].data[k] > JuMP.upper_bound(v)
                    if check1 || check2
                        _error("Parameter values violate parameter bounds.")
                    end
                end
            end
        end
    end
    return
end

# Update point variable info to consider the infinite variable
function _update_point_info(info::JuMP.VariableInfo, ivref::InfOptVariableRef)
    _update_point_info(info, ivref, Val(variable_type(ivref)))
    return
end

function _update_point_info(info::JuMP.VariableInfo,
                            ivref::InfOptVariableRef, ::Val{Infinite})
    if JuMP.has_lower_bound(ivref) && !info.has_fix && !info.has_lb
        info = JuMP.VariableInfo(true, JuMP.lower_bound(ivref),
                                 info.has_ub, info.upper_bound,
                                 info.has_fix, info.fixed_value,
                                 info.has_start, info.start,
                                 info.binary, info.integer)
    end
    if JuMP.has_upper_bound(ivref) && !info.has_fix && !info.has_ub
        info = JuMP.VariableInfo(info.has_lb, info.lower_bound,
                                 true, JuMP.upper_bound(ivref),
                                 info.has_fix, info.fixed_value,
                                 info.has_start, info.start,
                                 info.binary, info.integer)
    end
    if JuMP.is_fixed(ivref) && !info.has_fix  && !info.has_lb  && !info.has_ub
        info = JuMP.VariableInfo(info.has_lb, info.lower_bound,
                                 info.has_ub, info.upper_bound,
                                 true, JuMP.fix_value(ivref),
                                 info.has_start, info.start,
                                 info.binary, info.integer)
    end
    if !(JuMP.start_value(ivref) === NaN) && !info.has_start
        info = JuMP.VariableInfo(info.has_lb, info.lower_bound,
                                 info.has_ub, info.upper_bound,
                                 info.has_fix, info.fixed_value,
                                 true, JuMP.start_value(ivref),
                                 info.binary, info.integer)
    end
    if JuMP.is_binary(ivref) && !info.integer
        info = JuMP.VariableInfo(info.has_lb, info.lower_bound,
                                 info.has_ub, info.upper_bound,
                                 info.has_fix, info.fixed_value,
                                 info.has_start, info.start,
                                 true, info.integer)
    end
    if JuMP.is_integer(ivref) && !info.binary
        info = JuMP.VariableInfo(info.has_lb, info.lower_bound,
                                 info.has_ub, info.upper_bound,
                                 info.has_fix, info.fixed_value,
                                 info.has_start, info.start,
                                 info.binary, true)
    end
    return info
end

"""
    JuMP.build_variable(_error::Function, info::JuMP.VariableInfo,
                        var_type::Symbol;
                        parameter_refs::Union{InfOptVariableRef,
                                              AbstractArray{<:InfOptVariableRef},
                                              Tuple, Nothing} = nothing,
                        infinite_variable_ref::Union{InfOptVariableRef,
                                                     Nothing} = nothing,
                        parameter_values::Union{Number, AbstractArray{<:Number},
                                                Tuple, Nothing} = nothing,
                        error::Union{Function, Nothing} = nothing,
                        extra_kw_args...)

Extend the [`JuMP.build_variable`](@ref) function to accomodate `InfiniteOpt`
variable types. Returns the appropriate variable Datatype (i.e.,
[`InfiniteVariable`](@ref), [`PointVariable`](@ref), and
[`GlobalVariable`](@ref)). Primarily this method is to be used internally by the
appropriate constructor macros [`@infinite_variable`](@ref),
[`@point_variable`](@ref), and [`@global_variable`](@ref). However, it can be
called manually to build `InfiniteOpt` variables. Errors if an unneeded keyword
argument is given or if the keywoard arguments are formatted incorrectly (e.g.,
`parameter_refs` contains repeated parameter references when an infinite variable
is defined). Also errors if needed kewword arguments are negated.

**Examples**
```julia
julia> @infinite_parameter(m, 0 <= t <= 1)
t

julia> info = VariableInfo(false, 0, false, 0, false, 0, false, 0, false, false);

julia> inf_var = build_variable(error, info, Infinite, parameter_refs = t)
InfiniteVariable{Int64,Int64,Int64,Int64}(VariableInfo{Int64,Int64,Int64,Int64}
(false, 0, false, 0, false, 0, false, 0, false, false), (t,))

julia> ivref = add_variable(m, inf_var, "var_name")
var_name(t)

julia> pt_var = build_variable(error, info, Point, infinite_variable_ref = ivref,
                               parameter_values = 0.5)
PointVariable{Int64,Int64,Int64,Float64}(VariableInfo{Int64,Int64,Int64,Float64}
(false, 0, false, 0, false, 0, true, 0.0, false, false), var_name(t), (0.5,))

julia> gb_var = build_variable(error, info, Global)
GlobalVariable{Int64,Int64,Int64,Int64}(VariableInfo{Int64,Int64,Int64,Int64}
(false, 0, false, 0, false, 0, false, 0, false, false))
```
"""
function JuMP.build_variable(_error::Function, info::JuMP.VariableInfo,
                             var_type::Symbol;
                             parameter_refs::Union{InfOptVariableRef,
                                                   AbstractArray{<:InfOptVariableRef},
                                                   Tuple, Nothing} = nothing,
                             infinite_variable_ref::Union{InfOptVariableRef,
                                                          Nothing} = nothing,
                             parameter_values::Union{Number,
                                                     AbstractArray{<:Number},
                                                     Tuple, Nothing} = nothing,
                             error::Union{Function, Nothing} = nothing,
                             extra_kw_args...)
    if error != nothing
        _error = error # replace with macro error function
    end
    for (kwarg, _) in extra_kw_args
        _error("Unrecognized keyword argument $kwarg")
    end
    if !(var_type in [Infinite, Point, Global])
        _error("Unrecognized variable type $var_type, should be Infinite, " *
               "Point, or Global.")
    end
    if !isa(infinite_variable_ref, Nothing) && variable_type(infinite_variable_ref) != Infinite
        _error("infinite_variable_ref must be nothing or refer to a variable " *
               " of Infinite type.")
    end
    if var_type != Infinite && parameter_refs != nothing
        _error("Can only use the keyword argument 'parameter_refs' with " *
               "infinite variables.")
    elseif var_type != Point && (infinite_variable_ref != nothing || parameter_values != nothing)
        _error("Can only use the keyword arguments 'infinite_var' and " *
               "'parameter_values' with point variables.")
    elseif var_type == Infinite
        if parameter_refs == nothing
            _error("Parameter references not specified, use the var(params...) " *
                   "syntax or the parameter_refs keyword argument.")
        end
        if !isa(parameter_refs, Tuple)
            parameter_refs = (parameter_refs, )
        end
        _check_parameter_tuple(_error, parameter_refs)
        parameter_refs = _make_formatted_tuple(parameter_refs)
        _check_tuple_groups(_error, parameter_refs)
        return InfiniteVariable(info, parameter_refs)
    elseif var_type == Point
        if parameter_values == nothing || infinite_variable_ref == nothing
            _error("Must specify the infinite variable and the values of its " *
                   "infinite parameters")
        end
        if !isa(parameter_values, Tuple)
            parameter_values = (parameter_values, )
        end
        parameter_values = _make_formatted_tuple(parameter_values)
        _check_tuple_shape(_error, infinite_variable_ref, parameter_values)
        _check_tuple_values(_error, infinite_variable_ref, parameter_values)
        info = _update_point_info(info, infinite_variable_ref)
        return PointVariable(info, infinite_variable_ref, parameter_values)
    else
        return GlobalVariable(info)
    end
end

# Used to update the model.param_to_vars field
function _update_param_var_mapping(vref::InfOptVariableRef, prefs::Tuple)
    _update_param_var_mapping(vref, Val(variable_type(vref)), prefs)
    return
end

function _update_param_var_mapping(vref::InfOptVariableRef, ::Val{Infinite}, prefs::Tuple)
    model = JuMP.owner_model(vref)
    pref_list = _list_parameter_refs(prefs)
    for pref in pref_list
        if haskey(model.param_to_vars, JuMP.index(pref))
            push!(model.param_to_vars[JuMP.index(pref)], JuMP.index(vref))
        else
            model.param_to_vars[JuMP.index(pref)] = [JuMP.index(vref)]
        end
    end
    return
end

# check the pref tuple contains only valid parameters
function _check_parameters_valid(model::InfiniteModel, prefs::Tuple)
    pref_list = _list_parameter_refs(prefs)
    for pref in pref_list
        !JuMP.is_valid(model, pref) && error("Invalid Parameter reference " *
                                             "provided.")
    end
    return
end

# Used to add point variable support to parameter supports if necessary
function _update_param_supports(inf_vref::InfOptVariableRef,
                                param_values::Tuple)
    _update_param_supports(inf_vref, Val(variable_type(inf_vref)), param_values)
    return
end

function _update_param_supports(inf_vref::InfOptVariableRef, ::Val{Infinite},
                                param_values::Tuple)
    prefs = parameter_refs(inf_vref)
    for i in eachindex(prefs)
        if variable_type(prefs[i]) == Parameter
            add_supports(prefs[i], param_values[i])
        else
            for (k, v) in prefs[i].data
                add_supports(v, param_values[i].data[k])
            end
        end
    end
    return
end

# Used to update mapping infinite_to_points
function _update_infinite_point_mapping(pvref::InfOptVariableRef,
                                        ivref::InfOptVariableRef)
    _update_infinite_point_mapping(pvref, Val(variable_type(pvref)),
                                   ivref, Val(variable_type(ivref)))
    return
end

function _update_infinite_point_mapping(pvref::InfOptVariableRef, ::Val{Point}
                                        ivref::InfOptVariableRef, ::Val{Infinite})
    model = JuMP.owner_model(pvref)
    if haskey(model.infinite_to_points, JuMP.index(ivref))
        push!(model.infinite_to_points[JuMP.index(ivref)], JuMP.index(pvref))
    else
        model.infinite_to_points[JuMP.index(ivref)] = [JuMP.index(pvref)]
    end
    return
end

"""
    JuMP.add_variable(model::InfiniteModel, v::InfOptVariable, name::String = "")

Extend the [`JuMP.add_variable`](@ref) function to accomodate `InfiniteOpt`
variable types. Adds a variable to an infinite model `model` and returns an
appropriate variable reference (i.e., [`InfiniteVariableRef`](@ref),
[`PointVariableRef`](@ref), or [`GlobalVariableRef`](@ref)). Primarily intended
to be an internal function of the constructor macros [`@infinite_variable`](@ref),
[`@point_variable`](@ref), and [`@global_variable`](@ref). However, it can be used
in combination with [`JuMP.build_variable`](@ref) to add variables to an infinite
model object. Errors if invalid parameters reference(s) or an invalid infinite
variable reference is included in `v`.

**Examples**
```julia
julia> inf_var = build_variable(error, info, Infinite, parameter_refs = t);

julia> ivref = add_variable(m, inf_var, "var_name")
var_name(t)

julia> pt_var = build_variable(error, info, Point, infinite_variable_ref = ivref,
                               parameter_values = 0.5);

julia> pvref = add_variable(m, pt_var, "var_alias")
var_alias

julia> gb_var = build_variable(error, info, Global)

julia> gvref = add_variable(m, gb_var, "var_name")
var_name
```
"""
function JuMP.add_variable(model::InfiniteModel, v::InfOptVariable,
                           name::String = "")
    model.next_var_index += 1
    if isa(v, InfiniteVariable)
        _check_parameters_valid(model, v.parameter_refs)
        vref = InfOptVariableRef(model, model.next_var_index, Infinite)
        _update_param_var_mapping(vref, v.parameter_refs)
    elseif isa(v, PointVariable)
        ivref = v.infinite_variable_ref
        !JuMP.is_valid(model, ivref) && error("Invalid infinite variable " *
                                              "reference.")
        vref = InfOptVariableRef(model, model.next_var_index, Point)
        _update_param_supports(ivref, v.parameter_values)
        _update_infinite_point_mapping(vref, ivref)
    else
        vref = InfOptVariableRef(model, model.next_var_index, Global)
    end
    model.vars[JuMP.index(vref)] = v
    JuMP.set_name(vref, name)
    if v.info.has_lb
        newset = MOI.GreaterThan(convert(Float64, v.info.lower_bound))
        cref = JuMP.add_constraint(JuMP.owner_model(vref),
                                   JuMP.ScalarConstraint(vref, newset))
        _set_lower_bound_index(vref, JuMP.index(cref))
        model.constr_in_var_info[JuMP.index(cref)] = true
    end
    if v.info.has_ub
        newset = MOI.LessThan(convert(Float64, v.info.upper_bound))
        cref = JuMP.add_constraint(JuMP.owner_model(vref),
                                   JuMP.ScalarConstraint(vref, newset))
        _set_upper_bound_index(vref, JuMP.index(cref))
        model.constr_in_var_info[JuMP.index(cref)] = true
    end
    if v.info.has_fix
        newset = MOI.EqualTo(convert(Float64, v.info.fixed_value))
        cref = JuMP.add_constraint(model, JuMP.ScalarConstraint(vref, newset))
        _set_fix_index(vref, JuMP.index(cref))
        model.constr_in_var_info[JuMP.index(cref)] = true
    end
    if v.info.binary
        cref = JuMP.add_constraint(JuMP.owner_model(vref),
                                   JuMP.ScalarConstraint(vref, MOI.ZeroOne()))
        _set_binary_index(vref, JuMP.index(cref))
        model.constr_in_var_info[JuMP.index(cref)] = true
    elseif v.info.integer
        cref = JuMP.add_constraint(JuMP.owner_model(vref),
                                   JuMP.ScalarConstraint(vref, MOI.Integer()))
        _set_integer_index(vref, JuMP.index(cref))
        model.constr_in_var_info[JuMP.index(cref)] = true
    end
    model.var_in_objective[JuMP.index(vref)] = false
    return vref
end

"""
    JuMP.owner_model(vref::InfOptVariableRef)::InfiniteModel

Extend [`JuMP.owner_model`](@ref) function for `InfiniteOpt` variables. Returns
the infinite model associated with `vref`.

**Example**
```julia
julia> owner_model(vref)
An InfiniteOpt Model
Feasibility problem with:
Variable: 1
`GlobalVariableRef`-in-`MathOptInterface.LessThan{Float64}`: 1 constraint
`GlobalVariableRef`-in-`MathOptInterface.EqualTo{Float64}`: 1 constraint
`GlobalVariableRef`-in-`MathOptInterface.GreaterThan{Float64}`: 1 constraint
Names registered in the model: vref
Optimizer model backend information:
Model mode: AUTOMATIC
CachingOptimizer state: NO_OPTIMIZER
Solver name: No optimizer attached.
```
"""
JuMP.owner_model(vref::InfOptVariableRef)::InfiniteModel = vref.model

"""
    JuMP.index(v::InfOptVariableRef)::Int

Extent [`JuMP.index`](@ref) to return the index of a `InfiniteOpt` variable.

**Example**
```julia
julia> index(vref)
1
```
"""
JuMP.index(v::InfOptVariableRef)::Int64 = v.index

# used_by_constraint for variables
function used_by_constraint(vref::InfOptVariableRef, <:Variables)::Bool
    return haskey(JuMP.owner_model(vref).var_to_constrs, JuMP.index(vref))
end

# used_by_measure for variables
function used_by_measure(vref::InfOptVariableRef, <:Variables)::Bool
    return haskey(JuMP.owner_model(vref).var_to_meas, JuMP.index(vref))
end

"""
    used_by_objective(ref::InfOptVariableRef)::Bool

Return a `Bool` indicating if `ref` is used by the objective.

**Example**
```julia
julia> used_by_objective(vref)
true
```
"""
function used_by_objective(ref::InfOptVariableRef)::Bool
    return used_by_objective(ref, Val(variable_type(ref)))
end

function used_by_objective(vref::InfOptVariableRef, <:Variables)::Bool
    return JuMP.owner_model(vref).var_in_objective[JuMP.index(vref)]
end

# is_used for point and global variables
function is_used(vref::InfOptVariableRef, <:Variables)::Bool
    return used_by_measure(vref) || used_by_constraint(vref) || used_by_objective(vref)
end

# is_used for infinite variables
function is_used(vref::InfOptVariableRef, ::Val{Infinite})::Bool
    if used_by_measure(vref) || used_by_constraint(vref)
        return true
    end
    if used_by_point_variable(vref)
        for vindex in JuMP.owner_model(vref).infinite_to_points[JuMP.index(vref)]
            if is_used(InfOptVariableRef(JuMP.owner_model(vref), vindex, Point))
                return true
            end
        end
    end
    if used_by_reduced_variable(vref)
        for rindex in JuMP.owner_model(vref).infinite_to_reduced[JuMP.index(vref)]
            rvref = InfOptVariableRef(JuMP.owner_model(vref), rindex, Reduced)
            if used_by_constraint(rvref) || used_by_measure(rvref)
                return true
            end
        end
    end
    return false
end

"""
    used_by_point_variable(vref::InfOptVariableRef)::Bool

Return a `Bool` indicating if `vref` is used by a point variable.

**Example**
```julia
julia> used_by_point_variable(vref)
false
```
"""
function used_by_point_variable(vref::InfOptVariableRef)::Bool
    return used_by_point_variable(vref, Val(variable_type(vref)))
end

function used_by_point_variable(vref::InfOptVariableRef, ::Val{Infinite})::Bool
    return haskey(JuMP.owner_model(vref).infinite_to_points, JuMP.index(vref))
end

"""
    used_by_reduced_variable(vref::InfOptVariableRef)::Bool

Return a `Bool` indicating if `vref` is used by a reduced infinite variable.

**Example**
```julia
julia> used_by_reduced_variable(vref)
true
```
"""
function used_by_reduced_variable(vref::InfOptVariableRef)::Bool
    return used_by_reduced_variable(vref, Val(variable_type(vref)))
end

function used_by_reduced_variable(vref::InfOptVariableRef, ::Val{Infinite})::Bool
    return haskey(JuMP.owner_model(vref).infinite_to_reduced, JuMP.index(vref))
end


"""
    JuMP.delete(model::InfiniteModel, vref::InfOptVariableRef, <:Variables)

Extend [`JuMP.delete`](@ref) to delete `InfiniteOpt` variables and their
dependencies. Errors if variable is invalid, meaning it has already been
deleted or it belongs to another model.

**Example**
```julia
julia> print(model)
Min measure(g(t)*t) + z
Subject to
 z >= 0.0
 g(t) + z >= 42.0
 g(0.5) == 0
 t in [0, 6]

julia> delete(model, g)

julia> print(model)
Min measure(t) + z
Subject to
 z >= 0.0
 z >= 42.0
 t in [0, 6]
```
"""
# JuMP.delete for variables
function JuMP.delete(model::InfiniteModel, vref::InfOptVariableRef, <:Variables)
    @assert JuMP.is_valid(model, vref) "Variable is invalid."
    # update the optimizer model status
    if is_used(vref)
        set_optimizer_model_ready(model, false)
    end
    # remove variable info constraints associated with vref
    if JuMP.has_lower_bound(vref)
        JuMP.delete_lower_bound(vref)
    end
    if JuMP.has_upper_bound(vref)
        JuMP.delete_upper_bound(vref)
    end
    if JuMP.is_fixed(vref)
        JuMP.unfix(vref)
    end
    if JuMP.is_binary(vref)
        JuMP.unset_binary(vref)
    elseif JuMP.is_integer(vref)
        JuMP.unset_integer(vref)
    end
    # remove dependencies from measures and update them
    if used_by_measure(vref)
        for mindex in model.var_to_meas[JuMP.index(vref)]
#            if isa(model.measures[mindex].func, InfOptVariableRef)
            if isa(Val(variable_type(model.measures[mindex].func)), Variables)
                model.measures[mindex] = Measure(zero(JuMP.AffExpr),
                                                 model.measures[mindex].data)
            else
                _remove_variable(model.measures[mindex].func, vref)
            end
            JuMP.set_name(InfOptVariableRef(model, mindex, MeasureRef),
                           _make_meas_name(model.measures[mindex]))
        end
        # delete mapping
        delete!(model.var_to_meas, JuMP.index(vref))
    end
    # remove dependencies from measures and update them
    if used_by_constraint(vref)
        for cindex in model.var_to_constrs[JuMP.index(vref)]
#            if isa(model.constrs[cindex].func, InfOptVariableRef)
            if isa(Val(variable_type(model.constrs[cindex].func)), Variables)
                model.constrs[cindex] = JuMP.ScalarConstraint(zero(JuMP.AffExpr),
                                                      model.constrs[cindex].set)
            else
                _remove_variable(model.constrs[cindex].func, vref)
            end
        end
        # delete mapping
        delete!(model.var_to_constrs, JuMP.index(vref))
    end
    # remove from objective if vref is in it
    if used_by_objective(vref)
#        if isa(model.objective_function, InfOptVariableRef)
        if isa(Val(variable_type(model.objective_function)), Variables)
            model.objective_function = zero(JuMP.AffExpr)
        else
            _remove_variable(model.objective_function, vref)
        end
    end
    # do specific updates if vref is infinite
    if variable_type(vref) == Infinite
        # update parameter mapping
        all_prefs = _list_parameter_refs(parameter_refs(vref))
        for pref in all_prefs
            filter!(e -> e != JuMP.index(vref),
                    model.param_to_vars[JuMP.index(pref)])
            if length(model.param_to_vars[JuMP.index(pref)]) == 0
                delete!(model.param_to_vars, JuMP.index(pref))
            end
        end
        # delete associated point variables and mapping
        if used_by_point_variable(vref)
            for index in model.infinite_to_points[JuMP.index(vref)]
                JuMP.delete(model, InfOptVariableRef(model, index, Point))
            end
            delete!(model.infinite_to_points, JuMP.index(vref))
        end
        # delete associated reduced variables and mapping
        if used_by_reduced_variable(vref)
            for index in model.infinite_to_reduced[JuMP.index(vref)]
                JuMP.delete(model, InfOptVariableRef(model, index, Reduced))
            end
            delete!(model.infinite_to_reduced, JuMP.index(vref))
        end
    end
    # update mappings if is point variable
#    if isa(vref, PointVariableRef)
    if variable_type(vref) == Point
        ivref = infinite_variable_ref(vref)
        filter!(e -> e != JuMP.index(vref),
                model.infinite_to_points[JuMP.index(ivref)])
        if length(model.infinite_to_points[JuMP.index(ivref)]) == 0
            delete!(model.infinite_to_points, JuMP.index(ivref))
        end
    end
    # delete the variable information
    delete!(model.var_in_objective, JuMP.index(vref))
    delete!(model.vars, JuMP.index(vref))
    delete!(model.var_to_name, JuMP.index(vref))
    return
end

# JuMP.is_valid for variables
function JuMP.is_valid(model::InfiniteModel, vref::InfOptVariableRef,
                       <:Variables)::Bool
    return (model === JuMP.owner_model(vref) && JuMP.index(vref) in keys(model.vars))
end

"""
    JuMP.num_variables(model::InfiniteModel)::Int

Extend [`JuMP.num_variables`](@ref) to return the number of `InfiniteOpt`
variables assigned to `model`.

**Example**
```julia
julia> num_variables(model)
3
```
"""
JuMP.num_variables(model::InfiniteModel)::Int = length(model.vars)

# Include all the extension functions for manipulating the properties associated
# with VariableInfo
include("variable_info.jl")

# JuMP.name for variables
function JuMP.name(vref::InfOptVariableRef, <:Variables)::String
    return JuMP.owner_model(vref).var_to_name[JuMP.index(vref)]
end

# JuMP.set_name for global variables
function JuMP.set_name(vref::InfOptVariableRef, ::Val{Global}, name::String)
    JuMP.owner_model(vref).var_to_name[JuMP.index(vref)] = name
    JuMP.owner_model(vref).name_to_var = nothing
    return
end

# JuMP.set_name for point variables
function JuMP.set_name(vref::InfOptVariableRef, ::Val{Point}, name::String)
    if length(name) == 0
        inf_var_ref = infinite_variable_ref(vref, Point)
        name = _root_name(inf_var_ref)
        # TODO do something about SparseAxisArrays (report array of values in order)
        values = JuMP.owner_model(vref).vars[JuMP.index(vref)].parameter_values
        if length(values) == 1
            name = string(name, "(", values[1], ")")
        else
            name = string(name, values)
        end
    end
    JuMP.owner_model(vref).var_to_name[JuMP.index(vref)] = name
    JuMP.owner_model(vref).name_to_var = nothing
    return
end

# JuMP.set_name for infinite variables
function JuMP.set_name(vref::InfOptVariableRef, ::Val{Infinite}, root_name::String)
    if length(root_name) == 0
        root_name = "noname"
    end
    prefs = parameter_refs(vref)
    param_names = _root_names(prefs)
    param_name_tuple = "("
    for i = 1:length(param_names)
        if i != length(param_names)
            param_name_tuple *= string(param_names[i], ", ")
        else
            param_name_tuple *= string(param_names[i])
        end
    end
    param_name_tuple *= ")"
    var_name = string(root_name, param_name_tuple)
    JuMP.owner_model(vref).var_to_name[JuMP.index(vref)] = var_name
    JuMP.owner_model(vref).name_to_var = nothing
    return
end

"""
    infinite_variable_ref(vref::InfOptVariableRef)::InfOptVariableRef

Return the `InfOptVariableRef` associated with the point variable `vref`.

**Example**
```julia
julia> infinite_variable_ref(vref)
T(t, x)
```
"""
function infinite_variable_ref(vref::InfOptVariableRef)::InfOptVariableRef
    return infinite_variable_ref(vref, Val(variable_type(vref)))
end

function infinite_variable_ref(vref::InfOptVariableRef, ::Val{Point})::InfOptVariableRef
    return JuMP.owner_model(vref).vars[JuMP.index(vref)].infinite_variable_ref
end

"""
    parameter_values(vref::InfOptVariableRef)::Tuple

Return the support point associated with the point variable `vref`.

**Example**
```julia
julia> parameter_values(vref)
(0, )
```
"""
function parameter_values(vref::InfOptVariableRef)::Tuple
    return parameter_values(vref, Val(variable_type(vref)))
end

function parameter_values(vref::InfOptVariableRef, ::Val{Point})::Tuple
    return JuMP.owner_model(vref).vars[JuMP.index(vref)].parameter_values
end

# Internal function used to change the parameter value tuple of a point variable
function _update_variable_param_values(vref::InfOptVariableRef, pref_vals::Tuple)
    _update_variable_param_values(vref, Val(variable_type(vref)), pref_vals)
    return
end

function _update_variable_param_values(vref::InfOptVariableRef, ::Val{Point},
                                       pref_vals::Tuple)
    info = JuMP.owner_model(vref).vars[JuMP.index(vref)].info
    ivref = JuMP.owner_model(vref).vars[JuMP.index(vref)].infinite_variable_ref
    JuMP.owner_model(vref).vars[JuMP.index(vref)] = PointVariable(info, ivref,
                                                                  pref_vals)
    return
end

# Get root name of infinite variable
function _root_name(vref::InfOptVariableRef)
    return _root_name(vref, Val(variable_type(vref)))
end

function _root_name(vref::InfOptVariableRef, ::Val{Infinite})
    name = JuMP.name(vref)
    return name[1:findfirst(isequal('('), name)-1]
end

"""
    parameter_refs(vref::InfOptVariableRef)::Tuple

Return the `ParameterRef`(s) associated with the infinite variable `vref`. This
is formatted as a Tuple of containing the parameter references as they inputted
to define `vref`.

**Example**
```julia
julia> parameter_refs(vref)
(t,   [2]  =  x[2]
  [1]  =  x[1])
```
"""
function parameter_refs(vref::InfOptVariableRef)
    return parameter_refs(vref, Val(variable_type(vref)))
end

function parameter_refs(vref::InfOptVariableRef, ::Val{Infinite})
    return JuMP.owner_model(vref).vars[JuMP.index(vref)].parameter_refs
end

# Internal function used to change the parameter reference tuple of an infinite
# variable
function _update_variable_param_refs(vref::InfOptVariableRef, prefs::Tuple)
    _update_variable_param_refs(vref, Val(variable_type(vref)), prefs)
    return
end

function _update_variable_param_refs(vref::InfOptVariableRef, ::Val{Infinite}, prefs::Tuple)
    info = JuMP.owner_model(vref).vars[JuMP.index(vref)].info
    JuMP.owner_model(vref).vars[JuMP.index(vref)] = InfiniteVariable(info, prefs)
    return
end

"""
    set_parameter_refs(vref::InfOptVariableRef, prefs::Tuple)

Specify a new parameter reference tuple `prefs` for the infinite variable `vref`.
Note each element must contain a single parameter reference or an array of
parameter references. Errors if a parameter is double specified or if an element
contains parameters with different group IDs.

**Example**
```julia
julia> set_parameter_refs(vref, (t, x))

julia> parameter_refs(vref)
(t,   [2]  =  x[2]
  [1]  =  x[1])
```
"""
function set_parameter_refs(vref::InfOptVariableRef, prefs::Tuple)
    set_parameter_refs(vref, Val(variable_type(vref)), prefs)
    return
end

function set_parameter_refs(vref::InfOptVariableRef, ::Val{Infinite}, prefs::Tuple)
    _check_parameter_tuple(error, prefs)
    prefs = _make_formatted_tuple(prefs)
    _check_tuple_groups(error, prefs)
    _update_variable_param_refs(vref, prefs)
    JuMP.set_name(vref, _root_name(vref))
    if is_used(vref)
        set_optimizer_model_ready(JuMP.owner_model(vref), false)
    end
    return
end

"""
    add_parameter_ref(vref::InfOptVariableRef,
                      pref::Union{InfOptVariableRef, AbstractArray{<:InfOptVariableRef}})

Add additional parameter reference or group of parameter references to be
associated with the infinite variable `vref`. Errors if the parameter references
are already added to the variable or if the added parameters have different
group IDs.

```julia
julia> name(vref)
T(t)

julia> add_parameter_ref(vref, x)

julia> name(vref)
T(t, x)
```
"""
function add_parameter_ref(vref::InfOptVariableRef,
                       pref::Union{InfOptVariableRef, AbstractArray{<:InfOptVariableRef}})
    add_parameter_ref(vref, Val(variable_type(vref)), pref)
    return
end

function add_parameter_ref(vref::InfOptVariableRef, ::Val{Infinite},
                           pref::Union{InfOptVariableRef, AbstractArray{<:InfOptVariableRef}})
    set_parameter_refs(vref, (parameter_refs(vref)..., pref))
    return
end

# Make a variable reference
function _make_variable_ref(model::InfiniteModel, index::Int)::InfOptVariableRef
    if isa(model.vars[index], InfiniteVariable)
        return InfOptVariableRef(model, index, Infinite)
    elseif isa(model.vars[index], PointVariable)
        return InfOptVariableRef(model, index, Point)
    else
        return InfOptVariableRef(model, index, Global)
    end
end

"""
    JuMP.variable_by_name(model::InfiniteModel,
                          name::String)::Union{InfOptVariableRef, Nothing}

Extend [`JuMP.variable_by_name`](@ref) for `InfiniteModel` objects. Return the
varaible reference assoociated with a variable name. Errors if multiple
variables have the same name. Returns nothing if no such name exists.

**Examples**
```julia
julia> variable_by_name(m, "var_name")
var_name

julia> variable_by_name(m, "fake_name")

```
"""
function JuMP.variable_by_name(model::InfiniteModel,
                               name::String)::Union{InfOptVariableRef, Nothing}
    if model.name_to_var === nothing
        # Inspired from MOI/src/Utilities/model.jl
        model.name_to_var = Dict{String, Int64}()
        for (var, var_name) in model.var_to_name
            if haskey(model.name_to_var, var_name)
                # -1 is a special value that means this string does not map to
                # a unique variable name.
                model.name_to_var[var_name] = -1
            else
                model.name_to_var[var_name] = var
            end
        end
    end
    index = get(model.name_to_var, name, nothing)
    if index isa Nothing
        return nothing
    elseif index == -1
        error("Multiple variables have the name $name.")
    else
        return _make_variable_ref(model, index)
    end
    return
end

"""
    JuMP.all_variables(model::InfiniteModel)::Vector{InfOptVariableRef}

Extend [`JuMP.all_variables`](@ref) to return a list of all the variable
references associated with `model`.

**Examples**
```julia
julia> all_variables(m)
4-element Array{GeneralVariableRef,1}:
 ivar(test, θ)
 ivar2(test, x)
 name
 z
```
"""
function JuMP.all_variables(model::InfiniteModel)::Vector{InfOptVariableRef}
    vrefs_list = Vector{InfOptVariableRef}(undef, JuMP.num_variables(model))
    indexes = sort([index for index in keys(model.vars)])
    counter = 1
    for index in indexes
        vrefs_list[counter] = _make_variable_ref(model, index)
        counter += 1
    end
    return vrefs_list
end
