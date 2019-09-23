# JuMP.name for measure
function JuMP.name(mref::InfOptVariableRef, ::Val{MeasureRef})::String
    return JuMP.owner_model(mref).meas_to_name[mref.index]
end

# JuMP.set_name for measure
function JuMP.set_name(mref::InfOptVariableRef, ::Val{MeasureRef}, name::String)
    JuMP.owner_model(mref).meas_to_name[JuMP.index(mref)] = name
    return
end

# JuMP.is_valid for measures
function JuMP.is_valid(model::InfiniteModel, mref::InfOptVariableRef,
                       ::Val{MeasureRef})::Bool
    return (model === JuMP.owner_model(mref) && JuMP.index(mref) in keys(model.measures))
end

# Parse the string for displaying a measure
function _make_meas_name(meas::Measure)::String
    return string(meas.data.name, "(", JuMP.function_string(JuMP.REPLMode,
                                                            meas.func), ")")
end

# Used to update the model.var_to_meas and model.param_tomes fields
# this is needed to update measures if variables are deleted
function _update_var_meas_mapping(vrefs::Vector{InfOptVariableRef},
                                  mindex::Int)
    for vref in vrefs
        model = JuMP.owner_model(vref)
        if isa(Val(variable_type(vref)), Variables)
            if haskey(model.var_to_meas, JuMP.index(vref))
                push!(model.var_to_meas[JuMP.index(vref)], mindex)
            else
                model.var_to_meas[JuMP.index(vref)] = [mindex]
            end
        elseif variable_type(vref) == Parameter
            if haskey(model.param_to_meas, JuMP.index(vref))
                push!(model.param_to_meas[JuMP.index(vref)], mindex)
            else
                model.param_to_meas[JuMP.index(vref)] = [mindex]
            end
        elseif variable_type(vref) == MeasureRef
            if haskey(model.meas_to_meas, JuMP.index(vref))
                push!(model.meas_to_meas[JuMP.index(vref)], mindex)
            else
                model.meas_to_meas[JuMP.index(vref)] = [mindex]
            end
        elseif variable_type(vref) == Reduced
            if haskey(model.reduced_to_meas, JuMP.index(vref))
                push!(model.reduced_to_meas[JuMP.index(vref)], mindex)
            else
                model.reduced_to_meas[JuMP.index(vref)] = [mindex]
            end
        end
    end
    return
end

## Used to add the measure index to param_to_meas for parameters that are used
## in the evaluation data
# DiscreteMeasureData
function _update_param_data_mapping(model::InfiniteModel,
                                    data::DiscreteMeasureData,
                                    mindex::Int)
    if haskey(model.param_to_meas, JuMP.index(data.parameter_ref))
        if !(mindex in model.param_to_meas[JuMP.index(data.parameter_ref)])
            push!(model.param_to_meas[JuMP.index(data.parameter_ref)], mindex)
        end
    else
        model.param_to_meas[JuMP.index(data.parameter_ref)] = [mindex]
    end
    return
end

# MultiDiscreteMeasureData
function _update_param_data_mapping(model::InfiniteModel,
                                    data::MultiDiscreteMeasureData,
                                    mindex::Int)
    for pref in data.parameter_ref
        if haskey(model.param_to_meas, JuMP.index(pref))
            if !(mindex in model.param_to_meas[JuMP.index(pref)])
                push!(model.param_to_meas[JuMP.index(pref)], mindex)
            end
        else
            model.param_to_meas[JuMP.index(pref)] = [mindex]
        end
    end
    return
end

# Fallback
function _update_param_data_mapping(model::InfiniteModel, data::T,
                                    mindex::Int) where {T <: AbstractMeasureData}
    @warn "Unable to map parameter dependence for measure data type $T. " *
          "Parameter deletion methods should not be used."
    return
end

"""
    add_measure(model::InfiniteModel, meas::Measure)::InfOptVariableRef

Add a measure to `model` and return the corresponding measure reference. This
operates in a manner similar to [`JuMP.add_variable`](@ref).
"""
function add_measure(model::InfiniteModel, meas::Measure)::InfOptVariableRef
    model.next_meas_index += 1
    index = model.next_meas_index
    JuMP.check_belongs_to_model(meas.func, model)
    vrefs = _all_function_variables(meas.func)
    _update_var_meas_mapping(vrefs, index)
    _update_param_data_mapping(model, meas.data, index)
    mref = InfOptVariableRef(model, model.next_meas_index, MeasureRef)
    model.measures[mref.index] = meas
    JuMP.set_name(mref, _make_meas_name(meas))
    model.meas_in_objective[index] = false
    return mref
end

# Set a default weight function
_w(t) = 1

"""
    DiscreteMeasureData(parameter_ref::InfOptVariableRef
                        coefficients::Vector{<:Number},
                        supports::Vector{<:Number}; name::String = "measure",
                        weight_function::Function = w(t) = 1)::DiscreteMeasureData

Returns a `DiscreteMeasureData` object that can be utilized to define
measures using [`measure`](@ref). This accepts input for a scalar (single)
parameter. Note that `name` is used for printing purposes and a description of
the other arguments is provided in the documentation for
[`DiscreteMeasureData`](@ref). Errors if supports are out bounds or an unequal
number of supports and coefficients are given.

**Example**
```julia
julia> data = DiscreteMeasureData(pref, [0.5, 0.5], [1, 2], name = "example")
DiscreteMeasureData(pref, [0.5, 0.5], [1, 2], "example", InfiniteOpt._w)
```
"""
function DiscreteMeasureData(parameter_ref::InfOptVariableRef,
                             coefficients::Vector{<:Number},
                             supports::Vector{<:Number};
                             name::String = "measure",
                             weight_function::Function = _w
                             )::DiscreteMeasureData
    return DiscreteMeasureData(parameter_ref, coefficients, supports, name,
                               weight_function)
end

"""
    DiscreteMeasureData(parameter_ref::AbstractArray{InfOptVariableRef},
                        coefficients::Vector{<:Number},
                        supports::Vector{<:AbstractArray{<:Number}};
                        name::String = "measure",
                        weight_function::Function = w(t) = 1
                        )::MultiDiscreteMeasureData

Returns a `MultiDiscreteMeasureData` object that can be utilized to
define measures using [`measure`](@ref). This accepts input for an array (multi)
parameter. The inner arrays in the supports vector need to match the formatting
of the array used for `parameter_ref`. Note that `name` is used for printing
purposes and a description of the other arguments is provided in the
documentation for [`MultiDiscreteMeasureData`](@ref). Errors if supports are out
bounds, an unequal number of supports and coefficients are given, the array
formats do not match, or the parameters have different group IDs.

**Example**
```julia
julia> data = DiscreteMeasureData(prefs, [0.5, 0.5], [[1, 1], [2, 2]], name = "example");

julia> typeof(data)
MultiDiscreteMeasureData
"""
function DiscreteMeasureData(parameter_ref::AbstractArray{InfOptVariableRef},
                             coefficients::Vector{<:Number},
                             supports::Vector{<:AbstractArray};
                             name::String = "measure",
                             weight_function::Function = _w
                             )::MultiDiscreteMeasureData
    supports = [convert(JuMPC.SparseAxisArray, s) for s in supports]
    parameter_ref = convert(JuMPC.SparseAxisArray, parameter_ref)
    return MultiDiscreteMeasureData(parameter_ref, coefficients, supports, name,
                                    weight_function)
end

"""
    measure_function(mref::InfOptVariableRef)::JuMP.AbstractJuMPScalar

Return the function associated with `mref`.
"""
function measure_function(mref::InfOptVariableRef)::JuMP.AbstractJuMPScalar
    return measure_function(mref, Val(variable_type(mref)))
end

function measure_function(mref::InfOptVariableRef,
                          ::Val{MeasureRef})::JuMP.AbstractJuMPScalar
    return JuMP.owner_model(mref).measures[JuMP.index(mref)].func
end

"""
    measure_data(mref::InfOptVariableRef)::AbstractMeasureData

Return the measure data associated with `mref`.
"""
function measure_data(mref::InfOptVariableRef)::AbstractMeasureData
    return measure_data(mref, Val(variable_type(mref)))
end

function measure_data(mref::InfOptVariableRef,
                      ::Val{MeasureRef})::AbstractMeasureData
    return JuMP.owner_model(mref).measures[JuMP.index(mref)].data
end

# Check a measure function for a particular parameter and return Bool
function _has_parameter(vrefs::Vector{InfOptVariableRef},
                        pref::InfOptVariableRef)::Bool
    if _has_variable(vrefs, pref)
        return true
    end
    model = JuMP.owner_model(pref)
    relavent_ivindices = model.param_to_vars[JuMP.index(pref)]
    relavent_ivrefs = [InfOptVariableRef(model, vindex, Infinite)
                       for vindex in relavent_ivindices]
    for ivref in relavent_ivrefs
        if _has_variable(vrefs, ivref)
            return true
        elseif used_by_reduced_variable(ivref)
            for index in model.infinite_to_reduced[JuMP.index(ivref)]
                if _has_variable(vrefs, InfOptVariableRef(model, index, Reduced))
                    return true
                end
            end
        end
    end
    return false
end

## Check if expr contains a parameter directly or via an infinite variable
# scalar pref
function _check_has_parameter(expr::JuMP.AbstractJuMPScalar,
                              pref::InfOptVariableRef)
    vrefs = _all_function_variables(expr)
    if !_has_parameter(vrefs, pref)
        error("Measure expression is not parameterized by the parameter " *
              "specified in the measure data.")
    end
    return
end

# array pref
function _check_has_parameter(expr::JuMP.AbstractJuMPScalar,
                              pref::JuMPC.SparseAxisArray{InfOptVariableRef})
    vrefs = _all_function_variables(expr)
    for key in keys(pref.data)
        if !_has_parameter(vrefs, pref.data[key])
            error("Measure expression is not parameterized by the parameter " *
                  "specified in the measure data.")
        end
    end
    return
end

# Parse the model pertaining to an expression
function _model_from_expr(expr::JuMP.AbstractJuMPScalar)
    all_vrefs = _all_function_variables(expr)
    if length(all_vrefs) > 0
        return JuMP.owner_model(all_vrefs[1])
    else
        return
    end
end

## Internal functions for adding measure data supports to the parameter supports
# scalar pref
function _add_supports_to_parameters(pref::InfOptVariableRef,
                                     supports::Vector{<:Number})
    add_supports(pref, supports)
    return
end

# array pref
function _add_supports_to_parameters(pref::JuMPC.SparseAxisArray{InfOptVariableRef},
                                     supports::Array{<:JuMPC.SparseAxisArray{<:Number}})
    for i = 1:length(supports)
        for key in keys(pref.data)
            add_supports(pref.data[key], supports[i].data[key])
        end
    end
    return
end

"""
    measure(expr::JuMP.AbstractJuMPScalar, data::AbstractMeasureData)::InfOptVariableRef

Return a measure reference that evaluates `expr` using according to `data`. This
is the preferred method for implementing measures which follow the form:
``\\int_{p \\in P} expr(p) w(p) dp`` where ``p`` is an infinite parameter (scalar
or vector) and ``w`` is the weight function. The measure data `data` determines
how the measure is to be evaluated. Typically, the [`DiscreteMeasureData`](@ref)
constructor can be used to for `data`. The variable expression `expr` can contain
`InfiniteOpt` variables, infinite parameters, other measure references (meaning
measures can be nested), and constants. Errors if `expr` does not contain
infinite variables, infinite parameters, or measure references. Also errors if
the measure parameter specified in `data` is not in `expr` and is not in any
the nested measure references. Typically, this is called inside of
[`JuMP.@expression`](@ref), [`JuMP.@objective`](@ref), and
[`JuMP.@constraint`](@ref) in a manner similar to `sum`. Note measures are not
explicitly evaluated until [`build_optimizer_model!`](@ref) is called.

**Example**
```julia
julia> tdata = DiscreteMeasureData(t, [0.5, 0.5], [1, 2], name = "name1");

julia> xdata = DiscreteMeasureData(xs, [0.5, 0.5], [[-1, -1], [1, 1]],
                                   name = "name2");

julia> constr_RHS = @expression(model, measure(g - s + 2, tdata) + s^2)
name1(g(t) - s + 2) + s²

julia> @objective(model, Min, measure(g - 1  + measure(T, xdata), tdata))
name1(g(t) - 1 + name2(T(t, x)))
```
"""
function measure(expr::JuMP.AbstractJuMPScalar,
                 data::AbstractMeasureData)::InfOptVariableRef
#    if !isa(expr, Union{InfiniteExpr, MeasureExpr, ParameterExpr})
    if !_is_infinite_expr(expr) && !_is_measure_expr(expr) && !_is_parameter_expr(expr)
        error("Expression must contain infinite variables, infinite " *
              "parameters, or measure references")
    end
    model = _model_from_expr(expr)
    if model == nothing
        error("Expression contains no variables.")
    end
    pref = data.parameter_ref
    _check_has_parameter(expr, pref)
    meas = Measure(expr, data)
    _add_supports_to_parameters(pref, data.supports)
    return add_measure(model, meas)
end

# used_by_constraint for measure
function used_by_constraint(mref::InfOptVariableRef, ::Val{MeasureRef})::Bool
    return haskey(JuMP.owner_model(mref).meas_to_constrs, JuMP.index(mref))
end

# used_by_measure for measure
function used_by_measure(mref::InfOptVariableRef, ::Val{MeasureRef})::Bool
    return haskey(JuMP.owner_model(mref).meas_to_meas, JuMP.index(mref))
end

function used_by_objective(mref::InfOptVariableRef, ::Val{MeasureRef})::Bool
    return JuMP.owner_model(mref).meas_in_objective[JuMP.index(mref)]
end

# is_used for measures
function is_used(mref::InfOptVariableRef, ::Val{MeasureRef})::Bool
    return used_by_measure(mref) || used_by_constraint(mref) || used_by_objective(mref)
end

#=
"""
    JuMP.delete(model::InfiniteModel, mref::InfOptVariableRef)

Extend [`JuMP.delete`](@ref) to delete measures. Errors if measure is invalid,
meaning it does not belong to the model or it has already been deleted.

**Example**
```julia
julia> print(model)
Min measure(g(t)*t) + z
Subject to
 z >= 0.0
 measure(g(t)) == 0
 g(t) + z >= 42.0
 g(0.5) == 0
 t in [0, 6]

julia> delete(model, meas)

julia> print(model)
Min z
Subject to
 z >= 0.0
 0 == 0
 g(t) + z >= 42.0
 g(0.5) == 0
 t in [0, 6]
```
"""
=#
# JuMP.delete for measures
function JuMP.delete(model::InfiniteModel, mref::InfOptVariableRef, ::Val{MeasureRef})
    @assert JuMP.is_valid(model, mref) "Invalid measure reference."
    # Reset the transcription status
    if is_used(mref)
        set_optimizer_model_ready(model, false)
    end
    # Remove from dependent measures if there are any
    if used_by_measure(mref)
        for mindex in model.meas_to_meas[JuMP.index(mref)]
            if isa(model.measures[mindex].func, InfOptVariableRef) &&
               variable_type(model.measures[mindex].func) == MeasureRef
                data = model.measures[mindex].data
                model.measures[mindex] = Measure(zero(JuMP.AffExpr), data)
            else
                _remove_variable(model.measures[mindex].func, mref)
            end
            JuMP.set_name(InfOptVariableRef(model, mindex, MeasureRef),
                          _make_meas_name(model.measures[mindex]))
        end
        delete!(model.meas_to_meas, JuMP.index(mref))
    end
    # Remove from dependent constraints if there are any
    if used_by_constraint(mref)
        for cindex in model.meas_to_constrs[JuMP.index(mref)]
            if isa(model.constrs[cindex].func, InfOptVariableRef) &&
               variable_type(model.constrs[cindex].func) == MeasureRef
                model.constrs[cindex] = JuMP.ScalarConstraint(zero(JuMP.AffExpr),
                                                      model.constrs[cindex].set)
            else
                _remove_variable(model.constrs[cindex].func, mref)
            end
        end
        delete!(model.meas_to_constrs, JuMP.index(mref))
    end
    # Remove from objective if used there
    if used_by_objective(mref)
        if isa(model.objective_function, InfOptVariableRef) &&
           variable_type(model.objective_function) == MeasureRef
            model.objective_function = zero(JuMP.AffExpr)
        else
            _remove_variable(model.objective_function, mref)
        end
    end
    # Update that the variables used by it are no longer used by it
    vrefs = _all_function_variables(measure_function(mref))
    for vref in vrefs
        if isa(Val(variable_type(vref)), Variables)
            filter!(e -> e != JuMP.index(mref),
                    model.var_to_meas[JuMP.index(vref)])
            if length(model.var_to_meas[JuMP.index(vref)]) == 0
                delete!(model.var_to_meas, JuMP.index(vref))
            end
        elseif variable_type(vref) == Parameter
            filter!(e -> e != JuMP.index(mref),
                    model.param_to_meas[JuMP.index(vref)])
            if length(model.param_to_meas[JuMP.index(vref)]) == 0
                delete!(model.param_to_meas, JuMP.index(vref))
            end
        elseif variable_type(vref) == MeasureRef
            filter!(e -> e != JuMP.index(mref),
                    model.meas_to_meas[JuMP.index(vref)])
            if length(model.meas_to_meas[JuMP.index(vref)]) == 0
                delete!(model.meas_to_meas, JuMP.index(vref))
            end
        elseif variable_type(vref) == Reduced
            filter!(e -> e != JuMP.index(mref),
                    model.reduced_to_meas[JuMP.index(vref)])
            if length(model.reduced_to_meas[JuMP.index(vref)]) == 0
                delete!(model.reduced_to_meas, JuMP.index(vref))
            end
        end
    end
    # delete remaining measure information
    delete!(model.meas_in_objective, JuMP.index(mref))
    delete!(model.measures, JuMP.index(mref))
    delete!(model.meas_to_name, JuMP.index(mref))
    return
end
