module EnsembleMCMC

using Distributions: logpdf

@kwdef struct DifferentialEvolutionSettings
	chains_per_ensemble::Int
	target_acceptance_prob::Float64
	stretch_parameter::Float64
end

struct Chain
	samples::Vector{Vector{Float64}}
	logpdfs::Vector{Float64}
	accepted::Vector{Bool}
end

struct Ensemble
	chains::Vector{Chain}
end

struct SamplerCache
	proposals::Matrix{Float64}
	proposal_logpdfs::Vector{Float64}
end

mutable struct DifferentialEvolutionState{F, D}
	log_likelihood::F
	prior_dists::D
	γ0::Float64
	ensembles::Tuple{Ensemble, Ensemble}
	cache::SamplerCache
end

function log_posterior(sample, log_likelihood, prior_dists)
	logp = 0.0

	for (x, dist) in zip(sample, prior_dists)
		logp += logpdf(dist, x)
	end

	logp += log_likelihood(sample)

	return logp
end


"""
Evaluates batched log posterior distribution when samples are in a vector of vectors
"""
function log_posterior_batch!(logps, samples::Vector, log_likelihood, prior_dists)
	Threads.@threads for i in eachindex(samples)
		logps[i] = log_posterior(samples[i], log_likelihood, prior_dists)
	end

	return logps
end

"""
Evaluates batched log posterior distribution when samples are in a matrix of (d, num_samples)
"""
function log_posterior_batch!(logps, samples::Matrix, log_likelihood, prior_dists)
	Threads.@threads for i in axes(samples, 2)
		logps[i] = log_posterior(@views(samples[:, i]), log_likelihood, prior_dists)
	end

	return logps
end

function log_posterior_batch(samples::Matrix, log_likelihood, prior_dists)
	(dim, num_samples) = samples.size
	logps = zeros(num_samples)
	return log_posterior_batch!(logps, samples, log_likelihood, prior_dists)
end

function log_posterior_batch(samples::Vector, log_likelihood, prior_dists)
	num_samples = length(samples)
	logps = zeros(num_samples)
	return log_posterior_batch!(logps, samples, log_likelihood, prior_dists)
end

function sample_distributions(dists, num_samples)
	# TODO: seed
	out = zeros(num_samples, length(dists))
	for (j, dist) in enumerate(dists)
		out[:, j] .= rand(dist, num_samples)
	end
	return collect(transpose(out))
end

function initialize_ensembles(settings::DifferentialEvolutionSettings, args...)
	return initialize_ensembles(settings.chains_per_ensemble, args...)
end

function initialize_ensembles(chains_per_ensemble, log_likelihood, prior_dists)
	# TODO: seed
	samples = sample_distributions(prior_dists, 2 * chains_per_ensemble)
	logpdfs = log_posterior_batch(samples, log_likelihood, prior_dists)

	chains = [
		Chain([samples[:, i]], [logpdfs[i]], [true]) for i in eachindex(logpdfs)
	]

	ensemble_1 = Ensemble(chains[1:chains_per_ensemble])
	ensemble_2 = Ensemble(chains[chains_per_ensemble+1:end])
	return ensemble_1, ensemble_2
end

function initialize_sampler(settings::DifferentialEvolutionSettings, log_likelihood, prior_dists)
	# Initialize ensembles, sampling from the prior distributions
	γ0 = 2.38 / sqrt(2 * length(prior_dists))
	ensembles = initialize_ensembles(settings.chains_per_ensemble, log_likelihood, prior_dists)

	# Prepare sampler state and caches
	d, n = length(prior_dists), 2 * settings.chains_per_ensemble
	cache = SamplerCache(zeros(d, n), zeros(n))
	state = DifferentialEvolutionState(log_likelihood, prior_dists, γ0, ensembles, cache)
	return state
end

function calc_acceptance_fraction(ensembles::Tuple{Ensemble, Ensemble})
	n_accept = 0
	n_total = 0

	for ensemble in ensembles
		for chain in ensemble.chains
			for accept in chain.accepted
				n_accept += accept
				n_total += 1
			end
		end
	end

	return n_accept / n_total
end

"""For a global index [1, 2*chains_per_ensemble], returns ensemble index [1,2] and chain index [1,chains_per_ensemble]"""
global_to_ensemble_index(global_index, chains_per_ensemble) = fldmod1(global_index, chains_per_ensemble)

function ensemble_to_global_index((ensemble_index, chain_index), chains_per_ensemble)
	return (ensemble_index - 1) * chains_per_ensemble + chain_index
end

function step!(state::DifferentialEvolutionState, settings::DifferentialEvolutionSettings)
	(;proposals, proposal_logpdfs) = state.cache
	(;chains_per_ensemble, stretch_parameter) = settings

	# propose new samples and coalesce into matrix
	for i_ensemble in 1:2
		if i_ensemble == 1
			ensemble, other = state.ensembles
		else
			other, ensemble = state.ensembles
		end

		for (i_chain, chain) in enumerate(ensemble.chains)
			sample = chain.samples[end]

			# Calculate jump scale
			z = randn() * stretch_parameter
			γ = state.γ0 * (1 + z)

			# Draw two samples from the other ensemble, without replacement
			# TODO: probably a better way to do this than rejection sampling?
			i_chain_other_1 = rand(1:chains_per_ensemble)
			i_chain_other_2 = i_chain_other_1
			while i_chain_other_2 == i_chain_other_1
				i_chain_other_2 = rand(1:chains_per_ensemble)
			end

			# Calculate new sample using global workspace
			# Trying to avoid intermediate allocations
			i_global = ensemble_to_global_index((i_ensemble, i_chain), chains_per_ensemble)
			X1 = other.chains[i_chain_other_1].samples[end]
			X2 = other.chains[i_chain_other_2].samples[end]
			for k in eachindex(sample)
				proposals[k, i_global] = sample[k] + γ * (X2[k] - X1[k])
			end
		end
	end
	
	# eval batched logpdf
	log_posterior_batch!(proposal_logpdfs, proposals, state.log_likelihood, state.prior_dists)

	# accept or reject
	for (i, logp) in enumerate(proposal_logpdfs)
		i_ensemble, i_chain = global_to_ensemble_index(i, settings.chains_per_ensemble)
		chain = state.ensembles[i_ensemble].chains[i_chain]

		prev_logp = chain.logpdfs[end]
		log_p_accept = min(0, logp - prev_logp)
		log_u = log(rand())
		
		if log_u < log_p_accept
			# accept
			push!(chain.samples, proposals[:, i])
			push!(chain.logpdfs, logp)
			push!(chain.accepted, true)
		else
			# reject
			push!(chain.samples, copy(chain.samples[end]))
			push!(chain.logpdfs, prev_logp)
			push!(chain.accepted, false)
		end
	end

	# Calculate global acceptance fraction to adapt proposal scale 
	acceptance_fraction = calc_acceptance_fraction(state.ensembles)
	state.γ0 = state.γ0 * sqrt(acceptance_fraction / settings.target_acceptance_prob)

	return state
end

end # module EnsembleMCMC
