# Test the optimizer model methods
@testset "Optimizer Model" begin
    m = InfiniteModel()
    # optimizer_model
    @testset "optimizer_model" begin
        @test isa(optimizer_model(m), Model)
    end
    # optimizer_model_ready
    @testset "optimizer_model_ready" begin
        @test !optimizer_model_ready(m)
        m.ready_to_optimize = true
        @test optimizer_model_ready(m)
    end
    # set_optimizer_model_ready
    @testset "set_optimizer_model_ready" begin
        @test isa(set_optimizer_model_ready(m, false), Nothing)
        @test !optimizer_model_ready(m)
    end
    # set_optimizer_model
    @testset "set_optimizer_model" begin
        @test isa(set_optimizer_model(m, Model()), Nothing)
        @test length(optimizer_model(m).ext) == 0
    end
    # optimizer_model_key
    @testset "optimizer_model_key" begin
        m = InfiniteModel()
        @test optimizer_model_key(m) == :TransData
        optimizer_model(m).ext[:extra] = 42
        @test_throws ErrorException optimizer_model_key(m)
    end
end

# Test JuMP extensions
@testset "JuMP Extensions" begin
    m = InfiniteModel()
    mockoptimizer = () -> MOIU.MockOptimizer(MOIU.UniversalFallback(MOIU.Model{Float64}()),
                                             eval_objective_value=false)
    # bridge_constraints
    @testset "JuMP.bridge_constraints" begin
        @test !bridge_constraints(m)
        set_optimizer(optimizer_model(m), mockoptimizer)
        @test bridge_constraints(m)
    end
    # add_bridge
    @testset "JuMP.add_bridge" begin
        # @test isa(add_bridge(m, TestBridge), Nothing)
        @test isa(add_bridge(m, MOI.Bridges.Variable.VectorizeBridge), Nothing)
    end
    # set_optimizer
    @testset "JuMP.set_optimizer" begin
        m2 = InfiniteModel()
        @test isa(set_optimizer(m2, mockoptimizer), Nothing)
        @test m2.optimizer_constructor == mockoptimizer
    end
    # set_silent
    @testset "JuMP.set_silent" begin
        m2 = InfiniteModel()
        @test set_silent(m2)
    end
    # unset_silent
    @testset "JuMP.unset_silent" begin
        m2 = InfiniteModel()
        @test !unset_silent(m2)
    end
    # set_time_limit_sec
    @testset "JuMP.set_time_limit_sec" begin
        m2 = InfiniteModel()
        @test set_time_limit_sec(m2, 100) == 100
    end
    # unset_time_limit_sec
    @testset "JuMP.unset_time_limit_sec" begin
        m2 = InfiniteModel()
        @test isa(unset_time_limit_sec(m2), Nothing)
    end
    # time_limit_sec
    @testset "JuMP.time_limit_sec" begin
        m2 = InfiniteModel()
        @test_throws ErrorException time_limit_sec(m2)
    end
    # set_parameter
    @testset "JuMP.set_parameter" begin
        m2 = InfiniteModel()
        @test set_parameter(m2, "setting", 42) == 42
    end
    # solver_name
    @testset "JuMP.solver_name" begin
        @test solver_name(m) == "Mock"
    end
    # backend
    @testset "JuMP.backend" begin
        @test backend(m) == backend(optimizer_model(m))
    end
    # mode
    @testset "JuMP.mode" begin
        @test JuMP.mode(m) == JuMP.mode(optimizer_model(m))
    end
    # solve
    @testset "solve" begin
        @test_throws ErrorException solve(m)
    end
end
