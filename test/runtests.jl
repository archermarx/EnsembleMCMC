using EnsembleMCMC
using Test
using Distributions
using LinearAlgebra
using Printf
using CairoMakie: Makie as mk

@testset "Log-pdf evaluation" begin
	priors = [Normal(0,1), Normal(0,1)]
	likelihood = MvNormal([0.0, 0.0], LinearAlgebra.Diagonal([1.0, 1.0]))
	log_likelihood = x -> logpdf(likelihood, x)

	sample = [0.0, 0.0]

	logp_calc = EnsembleMCMC.log_posterior(sample, log_likelihood, priors)
	logp_exp = 4 * logpdf(Normal(0,1), 0.0)
	@test logp_calc ≈ logp_exp

	priors = [Uniform(-1,1), Uniform(-1,1)]
	logp_calc = EnsembleMCMC.log_posterior(sample, log_likelihood, priors)
	@test logp_calc ≈ 2 * logpdf(Normal(0,1), 0.0) + 2 * logpdf(Uniform(-1, 1), 0.0)

	samples = [[0.5, 0.0], [0.0, 0.5]]
	logp_calc = EnsembleMCMC.log_posterior_batch(samples, log_likelihood, priors)
	for (i, s) in enumerate(samples)
		@test logp_calc[i] ≈ EnsembleMCMC.log_posterior(s, log_likelihood, priors)
	end

	samples_matrix = hcat(samples...)
	logp_calc_matrix = EnsembleMCMC.log_posterior_batch(samples_matrix, log_likelihood, priors)
	@test all(logp_calc .≈ logp_calc_matrix)
end

@testset "Prior samples" begin
	priors = [Uniform(0, 1), Uniform(1, 2), Uniform(2, 3)]
	num_samples = 100
	samples = EnsembleMCMC.sample_distributions(priors, num_samples)
	@test samples.size == (length(priors), num_samples)
	@test all(x -> 0 <= x <= 1, samples[1, :])
	@test all(x -> 1 <= x <= 2, samples[2, :])
	@test all(x -> 2 <= x <= 3, samples[3, :])
end

@testset "Initialization" begin
	priors = [Uniform(0,1), Uniform(0,1), Uniform(0,1)]
	likelihood = MvNormal([0.5, 0.5, 0.5], LinearAlgebra.Diagonal([1.0, 1.0, 1.0]))
	log_likelihood = x -> logpdf(likelihood, x)

	chains_per_ensemble = 5
	d, n = length(priors), chains_per_ensemble
	chains, ensembles = EnsembleMCMC.initialize_ensemble(chains_per_ensemble, log_likelihood, priors)

	@test length(ensembles) == 2
	@test length(chains) == 2 * chains_per_ensemble
	@test length(ensembles[1]) == chains_per_ensemble
	@test length(ensembles[2]) == chains_per_ensemble
	check_accepted = x -> length(x.accepted) == 1 && x.accepted[]
	check_samples = x -> length(x.samples) == 1 && length(x.samples[]) == length(priors)
	check_logpdfs = x -> length(x.logpdfs) == 1 && isfinite(x.logpdfs[])

	for ensemble in ensembles
		@test all(check_accepted, ensemble)
		@test all(check_samples, ensemble)
		@test all(check_logpdfs, ensemble)
	end

	# Check for pointer equality between global chain members and ensembles 
	for (chain, ens_chain) in zip(chains[1:chains_per_ensemble], ensembles[1])
		@test chain == ens_chain
	end
	for (chain, ens_chain) in zip(chains[chains_per_ensemble+1:end], ensembles[2])
		@test chain == ens_chain
	end

	settings = EnsembleMCMC.DifferentialEvolutionSettings(;
		chains_per_ensemble,
		target_acceptance_prob = 0.25,
		stretch_parameter = 0.5
	)
	state = EnsembleMCMC.initialize_sampler(settings, log_likelihood, priors)

	@test state.cache.proposals.size == (d, n)
	@test state.cache.proposal_logpdfs.size == (n,)
	@test length(state.cache.ensembles) == 2
	@test state.cache.mean_logpdfs.size == (2*n,)
	@test state.cache.outlier_mask.size == (2*n,)

	EnsembleMCMC.step!(state, settings)
end	

@testset "Linear regression" begin
	N = 100
	m = 1.25
	b = -0.4
	ξ = 0.1

	x = sort(rand(N))
	f(x, (m, b)) = @. m * x + b
	y = f(x, (m, b)) .+ (ξ * randn(N))
	distribution = MvNormal(y, ξ^2 * I(N))

	function log_likelihood(p)	
		y_calc = f(x, p)
		return logpdf(distribution, y_calc)
	end

	priors = [Uniform(-2, 2), Uniform(-2, 2)]

	settings = EnsembleMCMC.DifferentialEvolutionSettings(;
		chains_per_ensemble = 20,
		target_acceptance_prob = 0.25,
		stretch_parameter = 0.5
	)

	state = EnsembleMCMC.initialize_sampler(settings, log_likelihood, priors)

	# warmup run
	num_samples = 500
	for i in 1:num_samples
		EnsembleMCMC.step!(state, settings, move_outliers=(i % 100 == 0))
	end

	for chain in state.chains
		@test length(chain.samples) == num_samples + 1
		@test length(chain.logpdfs) == num_samples + 1
		@test length(chain.accepted) == num_samples + 1
	end

	last_samples = [c.samples[end] for c in state.chains]
	last_logpdfs = [c.logpdfs[end] for c in state.chains]

	EnsembleMCMC.clear_history!(state)

	for (i, chain) in enumerate(state.chains)
		@test length(chain.samples) == 1
		@test length(chain.logpdfs) == 1
		@test length(chain.accepted) == 1
		@test all(chain.samples[] .== last_samples[i])
		@test chain.logpdfs[] == last_logpdfs[i]
		@test chain.accepted[] == true
	end

	# proper sampling run
	for i in 1:num_samples
		EnsembleMCMC.step!(state, settings, move_outliers=false)
	end

	p_accept = EnsembleMCMC.calc_acceptance_fraction(state.chains)
	@test 0.2 <= p_accept <= 0.3

	line = "-" ^ 34
	@printf("%s\n  Acceptance percentage: %.2f%%\n", line, p_accept * 100)

	samples = Vector{Float64}[]
	logpdfs = Float64[]
	best_logp = -Inf
	best_sample = -1

	frac_burn = 0.25
	num_burn = floor(Int, num_samples * frac_burn)
	global_ind = 0
	for c in state.chains
		for (i, (sample, logp)) in enumerate(zip(c.samples, c.logpdfs))
			if i <= num_burn
				continue
			end
			global_ind += 1
			if logp > best_logp
				best_logp = logp
				best_sample = global_ind
			end
			push!(samples, sample)
			push!(logpdfs, logp)
		end
	end

	@test length(samples) == (num_samples + 1 - num_burn) * 2 * settings.chains_per_ensemble

	m_best, b_best = samples[best_sample]
	ms = [s[1] for s in samples]
	bs = [s[2] for s in samples]

	q = [0.025, 0.5, 0.975]
	m_lo, m_med, m_hi = quantile(ms, q)
	b_lo, b_med, b_hi = quantile(bs, q)

	m_mean = mean(ms)
	b_mean = mean(bs)

	@test m_lo <= m <= m_hi
	@test b_lo <= b <= b_hi

	println(line * "\n  Slope\n" * line)
	@printf(
		"Truth: %.3f\nBest: %.3f\nMedian: %.3f\nMean: %.3f\n95%% CI: [%.3f, %.3f]\n",
		m, m_best, m_med, m_mean, m_lo, m_hi
	)
	println(line * "\n  Intercept\n" * line)
	@printf(
		"Truth: %.3f\nBest: %.3f\nMedian: %.3f\nMean: %.3f\n95%% CI: [%.3f, %.3f]\n",
		b, b_best, b_med, b_mean, b_lo, b_hi
	)
	println(line)

	plot = true

	if plot
		fig = mk.Figure()
		ax = mk.Axis(fig[1,1])
		mk.scatter!(x, y, alpha=1.0)

		M = 100
		lines = zeros(N, M)
		for i in 1:M
			sample = rand(samples)
			lines[:, i] = f(x, sample)
		end

		qs = [quantile(lines[i, :], [0.025, 0.975]) for i in 1:N]
		q_lo = [_q[1] for _q in qs]
		q_hi = [_q[2] for _q in qs]

		mk.lines!(x, f(x, (m_best, b_best)), color=:black)
		mk.lines!(x, q_lo, color=:black, linestyle=:dash)
		mk.lines!(x, q_hi, color=:black, linestyle=:dash)
		mk.save("samples_linear.png", fig, dpi=200)
	end
end

