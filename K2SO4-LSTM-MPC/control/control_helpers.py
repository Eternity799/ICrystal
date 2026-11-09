import numpy as np
def shifted_seed(previous_rates, matrix, default_rate):
    if previous_rates is None:
        return np.full(matrix.shape[1], default_rate)
    shifted = np.asarray(previous_rates)[1:]
    if len(shifted) != len(matrix):
        raise ValueError('Previous plan must shrink by exactly one step')
    return (matrix.T @ shifted)/matrix.sum(axis=0)

def choose_action(candidate, temperature, constraints, decision_seconds, period=60.):
    """A feasible incumbent may be used after cooperative timeout; never call it convergence."""
    if candidate is None or not candidate['feasible'] or decision_seconds >= period:
        return 0., 'hold_fallback'
    rate = float(candidate['rates'][0])
    if not np.isfinite(rate) or not (constraints['min_rate_c_min']-1e-7 <= rate <= constraints['max_rate_c_min']+1e-7):
        return 0., 'hold_fallback'
    if not constraints['min_temperature_c']-1e-7 <= temperature+rate <= constraints['max_temperature_c']+1e-7:
        return 0., 'hold_fallback'
    return rate, 'feasible_incumbent'

def cooling_profile(nodes,start_temperature,constraints):
    nodes=np.asarray(nodes,dtype=float)
    if nodes.ndim!=2 or nodes.shape[1]!=2 or len(nodes)<2 or not np.isfinite(nodes).all():
        raise ValueError('Cooling profile requires finite [minute, temperature] nodes')
    if nodes[0,0]!=0 or nodes[-1,0]!=180 or np.any(np.diff(nodes[:,0])<=0):
        raise ValueError('Profile times must strictly increase from 0 to 180')
    if not np.isclose(nodes[0,1],start_temperature,rtol=0,atol=1e-7):raise ValueError('Profile initial temperature differs from PBM')
    rates=np.diff(nodes[:,1])/np.diff(nodes[:,0])
    if (np.any(nodes[:,1]<constraints['min_temperature_c']-1e-7) or np.any(nodes[:,1]>constraints['max_temperature_c']+1e-7)
        or np.any(rates<constraints['min_rate_c_min']-1e-7) or np.any(rates>constraints['max_rate_c_min']+1e-7)):
        raise ValueError('Cooling profile violates temperature/rate constraints')
    return np.interp(np.arange(181),nodes[:,0],nodes[:,1])
