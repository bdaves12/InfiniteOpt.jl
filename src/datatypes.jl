"""
    AbstractInfiniteSet

An abstract type for sets that characterize infinite parameters.
"""
abstract type AbstractInfiniteSet end

"""
    InfOptParameter{T <: AbstractInfiniteSet} <: JuMP.AbstractVariable

A DataType for storing core infinite parameter information.

**Fields**
- `set::T` The infinite set that characterizes the parameter.
- `supports::Vector{<:Number}` The support points used to discretize this
                               parameter.
- `independent::Bool` Is independent of other parameters that share its group ID
                      number.
"""
struct InfOptParameter{T <: AbstractInfiniteSet} <: JuMP.AbstractVariable
    set::T
    supports::Vector{<:Number}
    independent::Bool
end

"""
    InfOptVariable <: JuMP.AbstractVariable

An abstract type for infinite, point, and hold variables.
"""
abstract type InfOptVariable <: JuMP.AbstractVariable end

"""
    AbstractReducedInfo

An abstract type reduced variable information.
"""
abstract type AbstractReducedInfo end

"""
    AbstractMeasureData

An abstract type to define data for measures to define the behavior of
[`Measure`](@ref).
"""
abstract type AbstractMeasureData end

"""
    Measure{T <: JuMP.AbstractJuMPScalar, V <: AbstractMeasureData}

A DataType for measure abstractions.

**Fields**
- `func::T` Variable expression.
- `data::V` Data of the abstraction as described in a `AbstractMeasureData`
            subtype.
"""
struct Measure{T <: JuMP.AbstractJuMPScalar, V <: AbstractMeasureData}
    func::T
    data::V
end

"""
    AbstractDerivativeVariable

An abstract DataType for derivatives.
"""
abstract type AbstractDerivativeVariable end

"""
    InfiniteModel <: JuMP.AbstractModel

A DataType for storing all of the mathematical modeling information needed to
model an optmization problem with an infinite dimensional decision space.

**Fields**
- `next_meas_index::Int` Index - 1 of next measure.
- `measures::Dict{Int, Measure}` Measure indices to measure datatypes.
- `meas_to_name::Dict{Int, String}` Measure indices to names.
- `meas_to_constrs::Dict{Int, Vector{Int}}` Measure indices to dependent
                                            constraint indices.
- `meas_to_meas::Dict{Int, Vector{Int}}` Measure indices to dependent
                                         measure indices.
- `meas_in_objective::Dict{Int, Bool}` Measure indices to if used in objective.
- `integral_defaults::Dict{Symbol}` Default keyword argument settings for measures.
- `next_param_index::Int` Index - 1 of next infinite parameter.
- `next_param_id::Int` Index - 1 of the next infinite parameter group.
- `params::Dict{Int, InfOptParameter}` Infinite parameter indices to parameter
                                       datatype.
- `param_to_name::Dict{Int, String}` Infinite parameter indices to names.
- `name_to_param::Union{Dict{String, Int}, Nothing}` Names to infinite
                                                     parameters.
- `param_to_group_id::Dict{Int, Int}` Infinite parameter indices to group IDs.
- `param_to_constrs::Dict{Int, Vector{Int}}` Infinite parameter indices to list
                                             of dependent constraint indices.
- `param_to_meas::Dict{Int, Vector{Int}}` Infinite parameter indices to list
                                          of dependent measure indices.
- `param_to_vars::Dict{Int, Vector{Int}}` Infinite parameter indices to list
                                          of dependent variable indices.
- `param_to_derivs::Dict{Int, Vector{Int}}` Infinite parameter indices to list
                                            of dependent derivative variables.
- `next_var_index::Int` Index - 1 of next variable index.
- `vars::Dict{Int, Dict{Int, Union{InfOptVariable, ReducedVariable}}` Variable
                                                  indices to variable datatype.
- `var_to_name::Dict{Int, String}` Variable indices to names.
- `name_to_var::Union{Dict{String, Int}, Nothing}` Variable names to indices.
- `var_to_lower_bound::Dict{Int, Int}` Variable indices to lower bound index.
- `var_to_upper_bound::Dict{Int, Int}` Variable indices to upper bound index.
- `var_to_fix::Dict{Int, Int}` Variable indices to fix index.
- `var_to_zero_one::Dict{Int, Int}` Variable indices to binary index.
- `var_to_integrality::Dict{Int, Int}` Variable indices to integer index.
- `var_to_constrs::Dict{Int, Vector{Int}}` Variable indices to dependent
                                           constraint indices.
- `var_to_meas::Dict{Int, Vector{Int}}` Variable indices to dependent
                                        measure indices.
- `var_in_objective::Dict{Int, Bool}` Variable indices to if used in objective.
- `infinite_to_points::Dict{Int, Vector{Int}}` Infinite variable indices to
                                               dependent point variable indices.
- `infinite_to_reduced::Dict{Int, Vector{Int}}` Infinite variable indices to
                                               dependent reduced variable indices.
- `infinite_to_derivs::Dict{Int, Vector{Int}}` Infinite variable indices to
                                               dependent derivatives
- `has_hold_bounds::Bool` Have hold variables with bounds been added to the model
- `next_deriv_index::Int` Index - 1 of next constraint
- `derivs::Dict{Int, AbstractDerivativeVariable}` Derivatives added to the model
- `deriv_to_name::Dict{Int, String}` Derivative names for printing
- `deriv_to_lower_bound::Dict{Int, Int}` Derivatives to lower bound constraints
- `deriv_to_upper_bound::Dict{Int, Int}` Derivatives to upper bound constraints
- `deriv_to_constrs::Dict{Int, Vector{Int}}` Derivatives to dependent constraints
- `deriv_to_derivs::Dict{Int, Vector{Int}}` Derivatives to dependent derivatives
- `deriv_to_meas::Dict{Int, Vector{Int}}` Derivatives to dependent measures
- `reduced_to_constrs::Dict{Int, Vector{Int}}` Reduced variable indices to dependent
                                               constraint indices.
- `reduced_to_meas::Dict{Int, Vector{Int}}` Reduced variable indices to dependent
                                            measure indices.
- `reduced_variable::Dict{Int, AbstractReducedInfo}` Reduced variable indices to
                                                 reduced variable information.
- `next_constr_index::Int` Index - 1 of next constraint.
- `constrs::Dict{Int, JuMP.AbstractConstraint}` Constraint indices to constraint
                                                datatypes.
- `constr_to_name::Dict{Int, String}` Constraint indices to names.
- `name_to_constr::Union{Dict{String, Int}, Nothing}` Constraint names to
                                                      indices.
- `constr_in_var_info::Dict{Int, Bool}` Constraint indices to if related to
                                        variable information constraints.
- `objective_sense::MOI.OptimizationSense` Objective sense.
- `objective_function::JuMP.AbstractJuMPScalar` Finite scalar function.
- `obj_dict::Dict{Symbol, Any}` Store Julia symbols used with `InfiniteModel`
- `optimizer_constructor` MOI optimizer constructor (e.g., Gurobi.Optimizer).
- `optimizer_model::JuMP.Model` Model used to solve `InfiniteModel`
- `ready_to_optimize::Bool` Is the optimizer_model up to date.
- `ext::Dict{Symbol, Any}` Store arbitrary extension information.
"""
mutable struct InfiniteModel <: JuMP.AbstractModel
    # Measure Data
    next_meas_index::Int
    measures::Dict{Int, Measure}
    meas_to_name::Dict{Int, String}
    meas_to_constrs::Dict{Int, Vector{Int}}
    meas_to_meas::Dict{Int, Vector{Int}}
    meas_in_objective::Dict{Int, Bool}
    integral_defaults::Dict{Symbol}

    # Parameter Data
    next_param_index::Int
    next_param_id::Int
    params::Dict{Int, InfOptParameter}
    param_to_name::Dict{Int, String}
    name_to_param::Union{Dict{String, Int}, Nothing}
    param_to_group_id::Dict{Int, Int}
    param_to_constrs::Dict{Int, Vector{Int}}
    param_to_meas::Dict{Int, Vector{Int}}
    param_to_vars::Dict{Int, Vector{Int}}
    param_to_derivs::Dict{Int, Vector{Int}}

    # Variable data
    next_var_index::Int
    vars::Dict{Int, InfOptVariable}
    var_to_name::Dict{Int, String}
    name_to_var::Union{Dict{String, Int}, Nothing}
    var_to_lower_bound::Dict{Int, Int}
    var_to_upper_bound::Dict{Int, Int}
    var_to_fix::Dict{Int, Int}
    var_to_zero_one::Dict{Int, Int}
    var_to_integrality::Dict{Int, Int}
    var_to_constrs::Dict{Int, Vector{Int}}
    var_to_meas::Dict{Int, Vector{Int}}
    var_in_objective::Dict{Int, Bool}
    infinite_to_points::Dict{Int, Vector{Int}}
    infinite_to_reduced::Dict{Int, Vector{Int}}
    infinite_to_derivs::Dict{Int, Vector{Int}}
    has_hold_bounds::Bool

    # Derivative data
    next_deriv_index::Int
    derivs::Dict{Int, AbstractDerivativeVariable}
    deriv_to_name::Dict{Int, String}
    deriv_to_lower_bound::Dict{Int, Int}
    deriv_to_upper_bound::Dict{Int, Int}
    deriv_to_constrs::Dict{Int, Vector{Int}}
    deriv_to_derivs::Dict{Int, Vector{Int}}
    deriv_to_meas::Dict{Int, Vector{Int}}

    # Reduced infinite variables
    reduced_to_constrs::Dict{Int, Vector{Int}}
    reduced_to_meas::Dict{Int, Vector{Int}}
    reduced_to_derivs::Dict{Int, Vector{Int}}
    reduced_variable::Dict{Int, <:AbstractReducedInfo}

    # Constraint Data
    next_constr_index::Int
    constrs::Dict{Int, JuMP.AbstractConstraint}
    constr_to_name::Dict{Int, String}
    name_to_constr::Union{Dict{String, Int}, Nothing}
    constr_in_var_info::Dict{Int, Bool}

    # Objective Data
    objective_sense::MOI.OptimizationSense
    objective_function::JuMP.AbstractJuMPScalar

    # Objects
    obj_dict::Dict{Symbol, Any}

    # Optimize Data
    optimizer_constructor
    optimizer_model::JuMP.Model
    ready_to_optimize::Bool

    # Extensions
    ext::Dict{Symbol, Any}
end

"""
    InfiniteModel([optimizer_constructor; seed::Bool = false,
                  OptimizerModel::Function = TranscriptionModel,
                  caching_mode::MOIU.CachingOptimizerMode = MOIU.AUTOMATIC,
                  bridge_constraints::Bool = true, optimizer_model_kwargs...])

Return a new infinite model where an optimizer is specified if an
`optimizer_constructor` is given. The `seed` argument indicates if the stochastic
sampling used in conjunction with the model should be seeded. The optimizer
can also later be set with the [`JuMP.set_optimizer`](@ref) call. By default
the `optimizer_model` data field is initialized with a
[`TranscriptionModel`](@ref), but a different type of model can be assigned via
[`set_optimizer_model`](@ref) as can be required by extensions.

**Example**
```jldoctest
julia> using InfiniteOpt, JuMP, Ipopt;

julia> model = InfiniteModel()
An InfiniteOpt Model
Feasibility problem with:
Variables: 0
Optimizer model backend information:
Model mode: AUTOMATIC
CachingOptimizer state: NO_OPTIMIZER
Solver name: No optimizer attached.

julia> model = InfiniteModel(Ipopt.Optimizer)
An InfiniteOpt Model
Feasibility problem with:
Variables: 0
Optimizer model backend information:
Model mode: AUTOMATIC
CachingOptimizer state: EMPTY_OPTIMIZER
Solver name: Ipopt
```
"""
function InfiniteModel(; seed::Bool = false,
                       OptimizerModel::Function = TranscriptionModel, kwargs...)
    if seed
        Random.seed!(0)
    end
    return InfiniteModel(# Measures
                         0, Dict{Int, Measure}(), Dict{Int, String}(),
                         Dict{Int, Vector{Int}}(), Dict{Int, Vector{Int}}(),
                         Dict{Int, Bool}(),
                         Dict(:eval_method => sampling,
                              :num_supports => 10,
                              :weight_func => _w,
                              :name => "integral",
                              :use_existing_supports => false),
                         # Parameters
                         0, 0, Dict{Int, InfOptParameter}(), Dict{Int, String}(),
                         nothing, Dict{Int, Int}(), Dict{Int, Vector{Int}}(),
                         Dict{Int, Vector{Int}}(), Dict{Int, Vector{Int}}(),
                         Dict{Int, Vector{Int}}(),
                         # Variables
                         0, Dict{Int, InfOptVariable}(),
                         Dict{Int, String}(), nothing, Dict{Int, Int}(),
                         Dict{Int, Int}(), Dict{Int, Int}(), Dict{Int, Int}(),
                         Dict{Int, Int}(), Dict{Int, Vector{Int}}(),
                         Dict{Int, Vector{Int}}(), Dict{Int, Bool}(),
                         Dict{Int, Vector{Int}}(), Dict{Int, Vector{Int}}(),
                         Dict{Int, Vector{Int}}(), false,
                         # Derivatives
                         0, Dict{Int, AbstractDerivativeVariable}(), Dict{Int, String}(),
                         Dict{Int, Int}(), Dict{Int, Int}(),
                         Dict{Int, Vector{Int}}(), Dict{Int, Vector{Int}}(),
                         Dict{Int, Vector{Int}}(),
                         # Placeholder variables
                         Dict{Int, Vector{Int}}(), Dict{Int, Vector{Int}}(),
                         Dict{Int, Vector{Int}}(), Dict{Int, ReducedInfiniteVariable}(),
                         # Constraints
                         0, Dict{Int, JuMP.AbstractConstraint}(),
                         Dict{Int, String}(), nothing, Dict{Int, Bool}(),
                         # Objective
                         MOI.FEASIBILITY_SENSE,
                         zero(JuMP.GenericAffExpr{Float64, FiniteVariableRef}),
                         # Object dictionary
                         Dict{Symbol, Any}(),
                         # Optimize data
                         nothing, OptimizerModel(;kwargs...), false,
                         # Extensions
                         Dict{Symbol, Any}())
end

## Set the optimizer_constructor depending on what it is
# MOI.OptimizerWithAttributes
function _set_optimizer_constructor(model::InfiniteModel,
                                    constructor::MOI.OptimizerWithAttributes)
    model.optimizer_constructor = constructor.optimizer_constructor
    return
end

# No attributes
function _set_optimizer_constructor(model::InfiniteModel, constructor)
    model.optimizer_constructor = constructor
    return
end

# Dispatch for InfiniteModel call with optimizer constructor
function InfiniteModel(optimizer_constructor; seed::Bool = false,
                       OptimizerModel::Function = TranscriptionModel, kwargs...)
    model = InfiniteModel(seed = seed)
    model.optimizer_model = OptimizerModel(optimizer_constructor; kwargs...)
    _set_optimizer_constructor(model, optimizer_constructor)
    return model
end

# Define basic InfiniteModel extensions
Base.broadcastable(model::InfiniteModel) = Ref(model)
JuMP.object_dictionary(model::InfiniteModel) = model.obj_dict

"""
    GeneralVariableRef <: JuMP.AbstractVariableRef

An abstract type for all variable references used with infinite models.
"""
abstract type GeneralVariableRef <: JuMP.AbstractVariableRef end

"""
    MeasureFiniteVariableRef <: GeneralVariableRef

An abstract type for inheritance of finite variable and measure references.
"""
abstract type MeasureFiniteVariableRef <: GeneralVariableRef end

"""
    FiniteVariableRef <: MeasureFiniteVariableRef

An abstract type of finite variable references (ones that do not need transcription).
"""
abstract type FiniteVariableRef <: MeasureFiniteVariableRef end

"""
    HoldVariableRef <: FiniteVariableRef

A DataType for finite fixed variable references (e.g., first stage variables,
steady-state variables). In other words, variables that are held constant over
the infinite domain or some sub-domain of the infinite domain. Such references
each point to an instance of [`HoldVariable`](@ref) stored in `model`.

**Fields**
- `model::InfiniteModel` Infinite model.
- `index::Int` Index of variable in model.
"""
struct HoldVariableRef <: FiniteVariableRef
    model::InfiniteModel
    index::Int
end

"""
    PointVariableRef <: FiniteVariableRef

A DataType for variables defined at a transcipted point (e.g., second stage
variable at a particular scenario, dynamic variable at a discretized time point).
Such references each point to an instance of [`PointVariable`](@ref) stored in
`model`.

**Fields**
- `model::InfiniteModel` Infinite model.
- `index::Int` Index of variable in model.
"""
struct PointVariableRef <: FiniteVariableRef
    model::InfiniteModel
    index::Int
end

"""
    InfiniteVariableRef <: GeneralVariableRef

A DataType for untranscripted infinite dimensional variable references (e.g.,
second stage variables, time dependent variables). Such references each point to
an instance of [`InfiniteVariable`](@ref) stored in `model`.

**Fields**
- `model::InfiniteModel` Infinite model.
- `index::Int` Index of variable in model.
"""
struct InfiniteVariableRef <: GeneralVariableRef
    model::InfiniteModel
    index::Int
end

"""
    ReducedInfiniteVariableRef <: GeneralVariableRef

A DataType for partially transcripted infinite dimensional variable references.
This is used to expand measures that contain infinite variables that are not
fully transcripted by the measure. Such references each point to an instance of
[`ReducedInfiniteVariable`](@ref) stored in `model`.

**Fields**
- `model::InfiniteModel` Infinite model.
- `index::Int` Index of variable in model.
"""
struct ReducedInfiniteVariableRef <: GeneralVariableRef
    model::InfiniteModel
    index::Int
end

"""
    ParameterRef <: GeneralVariableRef

A DataType for untranscripted infinite parameters references that parameterize
the infinite variables. Such references each point to an instance of
[`InfOptParameter`](@ref) stored in `model`.

**Fields**
- `model::InfiniteModel` Infinite model.
- `index::Int` Index of variable in model.
"""
struct ParameterRef <: GeneralVariableRef
    model::InfiniteModel
    index::Int
end

"""
    MeasureRef <: MeasureFiniteVariableRef

A DataType for referring to measures/integrals that are stored as [`Measure`](@ref)
objects in `model`.

**Fields**
- `model::InfiniteModel` Infinite model.
- `index::Int` Index of variable in model.
"""
struct MeasureRef <: MeasureFiniteVariableRef
    model::InfiniteModel
    index::Int
end

"""
    DerivativeVariableRef <: GeneralVariableRef

A DataType for referencing derivatives of infinite variables that are stored as
[`DerivativeVariable`](@ref) objects in `model`.

**Fields**
- `model::InfiniteModel` Infinite model.
- `index::Int` Index of variable in model.
"""
struct DerivativeVariableRef <: GeneralVariableRef
    model::InfiniteModel
    index::Int
end

"""
    ReducedDerivativeVariableRef <: GeneralVariableRef

A DataType for referencing derivatives of infinite variables whose infinite
parameter dependencies have been reduced and that are stored as
[`ReducedDerivativeVariable`](@ref) objects in `model`. This is analagous to
[`ReducedInfiniteVariableRef`](@ref) except for derivative variables.

**Fields**
- `model::InfiniteModel` Infinite model.
- `index::Int` Index of variable in model.
"""
struct ReducedDerivativeVariableRef <: GeneralVariableRef
    model::InfiniteModel
    index::Int
end

"""
    PointDerivativeVariableRef <: FiniteVariableRef

A DataType for referencing derivatives of infinite variables whose infinite
parameter dependencies have been fully transcribed and that are stored as
[`PointDerivativeVariable`](@ref) objects in `model`. This is analagous to
[`PointVariableRef`](@ref) except for derivatives variables.

**Fields**
- `model::InfiniteModel` Infinite model.
- `index::Int` Index of variable in model.
"""
struct PointDerivativeVariableRef <: FiniteVariableRef
    model::InfiniteModel
    index::Int
end

"""
    IntervalSet <: AbstractInfiniteSet

A DataType that stores the lower and upper interval bounds for infinite
parameters that are continuous over a certain that interval.

**Fields**
- `lower_bound::Float64` Lower bound of the infinite parameter.
- `upper_bound::Float64` Upper bound of the infinite parameter.
"""
struct IntervalSet <: AbstractInfiniteSet
    lower_bound::Float64
    upper_bound::Float64
    function IntervalSet(lower::Float64, upper::Float64)
        if lower > upper
            error("Invalid interval set bounds, lower bound is greater than " *
                  "upper bound.")
        end
        return new(lower, upper)
    end
end

"""
    IntervalSet(lower_bound::Number, upper_bound::Number)

A constructor for [`IntervalSet`](@ref) that converts values of type `Number` to
values of type `Float64` as required by `IntervalSet`.
"""
IntervalSet(lb::Number, ub::Number) = IntervalSet(convert(Float64, lb),
                                                  convert(Float64, ub))

"""
    DistributionSet{T <: Distributions.NonMatrixDistribution} <: AbstractInfiniteSet

A DataType that stores the distribution characterizing infinite parameters that
are random.

**Fields**
- `distribution::T` Distribution of the random parameter.
"""
struct DistributionSet{T <: Distributions.NonMatrixDistribution} <: AbstractInfiniteSet
    distribution::T
end

"""
    InfiniteVariable{S, T, U, V} <: InfOptVariable

A DataType for storing core infinite variable information. Note each element of
the parameter reference tuple must contain either a single
[`ParameterRef`](@ref) or an `AbstractArray` of `ParameterRef`s where each
`ParameterRef` has the same group ID number.

**Fields**
- `info::JuMP.VariableInfo{S, T, U, V}` JuMP variable information.
- `parameter_refs::Tuple` The infinite parameters(s) that parameterize the
                          variable.
"""
struct InfiniteVariable{S, T, U, V} <: InfOptVariable
    info::JuMP.VariableInfo{S, T, U, V}
    parameter_refs::Tuple
end

"""
    PointVariable{S, T, U, V} <: InfOptVariable

A DataType for storing point variable information. Note that the elements
`parameter_values` field must match the format of the parameter reference tuple
defined in [`InfiniteVariable`](@ref)

**Fields**
- `info::JuMP.VariableInfo{S, T, U, V}` JuMP Variable information.
- `infinite_variable_ref::InfiniteVariableRef` The infinite variable associated
                                               with the point variable.
- `parameter_values::Tuple` The infinite parameter values defining the point.
"""
struct PointVariable{S, T, U, V} <: InfOptVariable
    info::JuMP.VariableInfo{S, T, U, V}
    infinite_variable_ref::InfiniteVariableRef
    parameter_values::Tuple
end

## Modify parameter dictionary to expand any multidimensional parameter keys
# Case where dictionary is already in correct form
function _expand_parameter_dict(param_bounds::Dict{ParameterRef,
                                                   IntervalSet})::Dict
    return param_bounds
end

# Case where dictionary contains vectors
function _expand_parameter_dict(param_bounds::Dict{<:Any, IntervalSet})::Dict
    # Initialize new dictionary
    new_dict = Dict{ParameterRef, IntervalSet}()
    # Find vector keys and expand
    for (key, set) in param_bounds
        # expand over the array of parameters if this is
        if isa(key, AbstractArray)
            for param in values(key)
                new_dict[param] = set
            end
        # otherwise we have parameter reference
        else
            new_dict[key] = set
        end
    end
    return new_dict
end

# Case where dictionary contains vectors
function _expand_parameter_dict(param_bounds::Dict)
    error("Invalid parameter bound dictionary format.")
end

"""
    ParameterBounds

A DataType for storing intervaled bounds of parameters. This is used to define
subdomains of [`HoldVariable`](@ref)s and [`BoundedScalarConstraint`](@ref)s.

**Fields**
- `intervals::Dict{ParameterRef, IntervalSet}` A dictionary of parameter intervals
that are tighter than those already associated those paraticular parameters.
"""
struct ParameterBounds
    intervals::Dict{ParameterRef, IntervalSet}
    function ParameterBounds(intervals::Dict)
        return new(_expand_parameter_dict(intervals))
    end
end

# Default method
function ParameterBounds()
    return ParameterBounds(Dict{ParameterRef, IntervalSet}())
end

"""
    HoldVariable{S, T, U, V} <: InfOptVariable

A DataType for storing hold variable information.

**Fields**
- `info::JuMP.VariableInfo{S, T, U, V}` JuMP variable information.
- `parameter_bounds::ParameterBounds` Valid parameter sub-domains
"""
struct HoldVariable{S, T, U, V} <: InfOptVariable
    info::JuMP.VariableInfo{S, T, U, V}
    parameter_bounds::ParameterBounds
end

"""
    ReducedInfiniteVariable <: AbstractReducedInfo

A DataType for storing reduced infinite variable information.

**Fields**
- `infinite_variable_ref::InfiniteVariableRef` The original infinite variable.
- `eval_supports::Dict{Int, Union{Number, JuMP.Containers.SparseAxisArray{<:Number}}}`
  The original parameter tuple indices to the evaluation supports.
"""
struct ReducedInfiniteVariable <: AbstractReducedInfo
    infinite_variable_ref::InfiniteVariableRef
    eval_supports::Dict{Int, Union{Number, JuMPC.SparseAxisArray{<:Number}}}
end

"""
    DerivativeVariable{T <: GeneralVariableRef} <: AbstractDerivativeVariable

A DataType for storing derivative variables which track taking derivatives of
infinite, reduced infinite, and derivative variables with respect to an infinite
parameter.

**Fields**
- `info::JuMP.VariableInfo` JuMP variable information.
- `variable_ref::T` The dependent variable of the derivative.
- `parameter_ref::ParameterRef` The parameter that the derivative is with respect to.
"""
struct DerivativeVariable{T <: GeneralVariableRef} <: AbstractDerivativeVariable
    info::JuMP.VariableInfo
    variable_ref::T
    parameter_ref::ParameterRef
end

"""
    ReducedDerivativeVariable <: AbstractDerivativeVariable

A DataType for storing reduced derivative variable information.

**Fields**
- `derivative_ref::DerivativeVariableRef` The original derivative variable.
- `eval_supports::Dict{Int, Union{Number, JuMP.Containers.SparseAxisArray{<:Number}}}`
  The original parameter tuple indices to the evaluation supports.
"""
struct ReducedDerivativeVariable <: AbstractDerivativeVariable
    derivative_ref::DerivativeVariableRef
    eval_supports::Dict{Int, Union{Number, JuMPC.SparseAxisArray{<:Number}}}
end

"""
    PointDerivativeVariable <: AbstractDerivativeVariable

A DataType for storing point variable information. Note that the elements
`parameter_values` field must match the format of the parameter reference tuple
associated with the dependent [`DerivativeVariable`](@ref) object.

**Fields**
- `info::JuMP.VariableInfo{S, T, U, V}` JuMP Variable information.
- `derivative_ref::DerivativeVariableRef` The derivative variable associated
                                               with the point variable.
- `parameter_values::Tuple` The infinite parameter values defining the point.
"""
struct PointDerivativeVariable <: AbstractDerivativeVariable
    info::JuMP.VariableInfo
    derivative_ref::DerivativeVariableRef
    parameter_values::Tuple
end

"""
    InfOptVariableRef

A union type for infinite, point, and hold variable references.
"""
const InfOptVariableRef = Union{InfiniteVariableRef, PointVariableRef,
                                HoldVariableRef}

# Define infinite expressions
const InfiniteExpr = Union{InfiniteVariableRef, ReducedInfiniteVariableRef,
                           DerivativeVariableRef, ReducedDerivativeVariableRef,
                           JuMP.GenericAffExpr{Float64, InfiniteVariableRef},
                           JuMP.GenericAffExpr{Float64, GeneralVariableRef},
                           JuMP.GenericAffExpr{Float64, ReducedInfiniteVariableRef},
                           JuMP.GenericAffExpr{Float64, DerivativeVariableRef},
                           JuMP.GenericAffExpr{Float64, ReducedDerivativeVariableRef},
                           JuMP.GenericQuadExpr{Float64, InfiniteVariableRef},
                           JuMP.GenericQuadExpr{Float64, GeneralVariableRef},
                           JuMP.GenericQuadExpr{Float64, ReducedInfiniteVariableRef},
                           JuMP.GenericQuadExpr{Float64, DerivativeVariableRef},
                           JuMP.GenericQuadExpr{Float64, ReducedDerivativeVariableRef}}
const ParameterExpr = Union{ParameterRef,
                            JuMP.GenericAffExpr{Float64, ParameterRef},
                            JuMP.GenericQuadExpr{Float64, ParameterRef}}

"""
    DiscreteMeasureData <: AbstractMeasureData

A DataType for one dimensional measure abstraction data where the measure
abstraction is of the form:
``measure = \\int_{\\tau \\in T} f(\\tau) w(\\tau) d\\tau \\approx \\sum_{i = 1}^N \\alpha_i f(\\tau_i) w(\\tau_i)``.

**Fields**
- `parameter_ref::ParameterRef` The infinite parameter over which the
                                integration occurs.
- `coefficients::Vector{<:Number}` Coefficients ``\\alpha_i`` for the above
                                   measure abstraction.
- `supports::Vector{<:Number}` Support points ``\\tau_i`` for the above
                               measure abstraction.
- `name::String` Name of the measure that will be implemented.
- `weight_function::Function` Weighting function ``w`` must map support value
                              input value of type `Number` to a scalar value.
"""
struct DiscreteMeasureData <: AbstractMeasureData
    parameter_ref::ParameterRef
    coefficients::Vector{<:Number}
    supports::Vector{<:Number}
    name::String
    weight_function::Function
    # Constructor
    function DiscreteMeasureData(parameter_ref::ParameterRef,
                                 coeffs::Vector{<:Number},
                                 supports::Vector{<:Number},
                                 name::String, weight_func::Function)
        if length(coeffs) != length(supports)
            error("The amount of coefficients must match the amount of " *
                  "support points.")
        end
        set = infinite_set(parameter_ref)
        # set = parameter_ref.model.params[parameter_ref.index].set
        if !supports_in_set(supports, set)
            error("Support points violate parameter domain.")
        end
        return new(parameter_ref, coeffs, supports, name, weight_func)
    end
end

"""
    MultiDiscreteMeasureData <: AbstractMeasureData

A DataType for multi-dimensional measure abstraction data where the measure
abstraction is of the form:
``measure = \\int_{\\tau \\in T} f(\\tau) w(\\tau) d\\tau \\approx \\sum_{i = 1}^N \\alpha_i f(\\tau_i) w(\\tau_i)``.

**Fields**
- `parameter_ref::JuMP.Containers.SparseAxisArray{<:ParameterRef}` The infinite
   parameters over which the integration occurs.
- `coefficients::Vector{<:Number}` Coefficients ``\\alpha_i`` for the above
                                   measure abstraction.
- `supports::Vector{<:JuMP.Containers.SparseAxisArray{<:Number}}` Support points
   ``\\tau_i`` for the above measure abstraction.
- `name::String` Name of the measure that will be implemented.
- `weight_function::Function` Weighting function ``w`` must map a numerical
                              support of type `JuMP.Containers.SparseAxisArray`
                              to a scalar value.
"""
struct MultiDiscreteMeasureData <: AbstractMeasureData
    parameter_ref::JuMPC.SparseAxisArray{<:ParameterRef}
    coefficients::Vector{<:Number}
    supports::Vector{<:JuMPC.SparseAxisArray}
    name::String
    weight_function::Function
    # Constructor
    function MultiDiscreteMeasureData(parameter_ref::JuMPC.SparseAxisArray{<:ParameterRef},
                                      coeffs::Vector{<:Number},
                                      supports::Vector{<:JuMPC.SparseAxisArray},
                                      name::String, weight_func::Function)
        if length(coeffs) != length(supports)
            error("The amount of coefficients must match the amount of " *
                  "support points.")
        elseif !_only_one_group(parameter_ref)
            error("Parameters must all have same group ID.")
        elseif length(supports) == 0
            error("Must specify at least one support.")
        elseif keys(supports[1].data) != keys(parameter_ref.data)
            error("The keys/dimensions of the support points and parameters " *
                  "do not match.")
        end
        for key in keys(parameter_ref.data)
            set = infinite_set(parameter_ref.data[key])
            for supp in supports
                if !supports_in_set(supp.data[key], set)
                    error("Support points violate parameter domain.")
                end
            end
        end
        return new(parameter_ref, coeffs, supports, name, weight_func)
    end
end

# Define finite measure expressions (note infinite expression take precedence)
const MeasureExpr = Union{MeasureRef,
                          JuMP.GenericAffExpr{Float64, MeasureRef},
                          JuMP.GenericAffExpr{Float64, MeasureFiniteVariableRef},
                          JuMP.GenericQuadExpr{Float64, MeasureRef},
                          JuMP.GenericQuadExpr{Float64, MeasureFiniteVariableRef}}

"""
    BoundedScalarConstraint{F <: JuMP.AbstractJuMPScalar,
                            S <: MOI.AbstractScalarSet} <: JuMP.AbstractConstraint

A DataType that stores infinite constraints defined on a subset of the infinite
parameters on which they depend.

**Fields**
- `func::F` The JuMP object.
- `set::S` The MOI set.
- `bounds::ParameterBounds` Set of valid parameter sub-domains that further bound
                            constraint.
- `orig_bounds::ParameterBounds` Set of the constraint's original parameter
                                 sub-domains (not considering hold variables)
"""
struct BoundedScalarConstraint{F <: JuMP.AbstractJuMPScalar,
                               S <: MOI.AbstractScalarSet} <: JuMP.AbstractConstraint
    func::F
    set::S
    bounds::ParameterBounds
    orig_bounds::ParameterBounds
end

"""
    GeneralConstraintRef

An abstract type for constraint references unique to InfiniteOpt.
"""
abstract type GeneralConstraintRef end

"""
InfiniteConstraintRef{S <: JuMP.AbstractShape} <: GeneralConstraintRef

A DataType for constraints that contain infinite variables.

**Fields**
- `model::InfiniteModel` Infinite model.
- `index::Int` Index of constraint in model.
- `shape::JuMP.AbstractShape` Shape of constraint
"""
struct InfiniteConstraintRef{S <: JuMP.AbstractShape} <: GeneralConstraintRef
    model::InfiniteModel
    index::Int
    shape::S
end

"""
    FiniteConstraintRef{S <: JuMP.AbstractShape} <: GeneralConstraintRef

A DataType for constraints that contain finite variables.

**Fields**
- `model::InfiniteModel` Infinite model.
- `index::Int` Index of constraint in model.
- `shape::JuMP.AbstractShape` Shape of constraint
"""
struct FiniteConstraintRef{S <: JuMP.AbstractShape} <: GeneralConstraintRef
    model::InfiniteModel
    index::Int
    shape::S
end

"""
    MeasureConstraintRef{S <: JuMP.AbstractShape} <: GeneralConstraintRef

A DataType for constraints that contain finite variables and measures.

**Fields**
- `model::InfiniteModel` Infinite model.
- `index::Int` Index of constraint in model.
- `shape::JuMP.AbstractShape` Shape of constraint
"""
struct MeasureConstraintRef{S <: JuMP.AbstractShape} <: GeneralConstraintRef
    model::InfiniteModel
    index::Int
    shape::S
end
