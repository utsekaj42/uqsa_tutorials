import numpy as np
import chaospy as cp
import matplotlib.pyplot as plt

def calculate_uqsa_measures(joint_dist, polynomial, alpha=5):
    """ Use chaospy to calculate appropriate indices of uq and sa"""
    dists = joint_dist
    mean =cp.E(polynomial, dists) 
    var = cp.Var(polynomial, dists)
    std = cp.Std(polynomial, dists)
    conInt = cp.Perc(polynomial, [alpha/2., 100-alpha/2.], joint_dist)
    sens_m = cp.Sens_m(polynomial,dists)
    sens_m2 = cp.Sens_m2(polynomial,dists)
    sens_t = cp.Sens_t(polynomial,dists)
    return dict(mean=mean,var=var,std=std,conInt=conInt,sens_m=sens_m,sens_m2=sens_m2,sens_t=sens_t) 

def check_convergence(samples, data, joint_dist, max_order, norm_ord=None, ref_values=None,  zero_offset=0.0):
    """
    Args:
        samples := (first axis must be along sample index not input index)
        model_evals :=
        norm_ord := any of the orders of norms supported by numpy.linalg.norm 
        ref_values(dict) := a dictionary of reference values to compare each case
        max_order(int) :=

    Calculate and display convergence of error with respect to the mean, and random check values.
    1. Determine "maximum/reference" order
    2. calculate reference values for E, var, Sm, St
    3. For each order calculate the error wrt to the reference value for all possible sample factors
    3.a. The results will be stored in a dict like 
        {measure: 
            {order: 
                [[sample_sizes], 
                [error_value]]
            }
        }
    4. Once complete plot the convergence of each measure
        fig_idxs = {measure:idx for idx,measure in enumerate(errors_dict.keys())}
        for measure, value in errors_dict.iteritems():
            plt.figure(fig_idxs[measure])
            for order,data in value.iteritems():

            plt.plot(value[
    """

    orders = range(1, max_order)
    measures = ["mean", "var"]
    n_pts_per_order = 10
    if len(data.shape)>1:
        model_dim = data.shape[-1] #TODO this is only valid for 1D arrays
    else:
        model_dim = 1
    error_dim = 3
    input_dim = len(joint_dist)

    error_dict = {measure:{order:np.zeros((error_dim,n_pts_per_order)) for order in orders} for measure in measures}

    submeasures = ["sens_m", "sens_t"]

    suberror_dict = {submeasure:{par:{order:np.zeros((error_dim,n_pts_per_order)) for order in orders} for par in range(input_dim)} for submeasure in submeasures}

    residual_error_dict = {idx:{order:np.zeros((error_dim,n_pts_per_order)) for order in orders} for idx in range(model_dim)}
    raw_sensitivities = {submeasure:{par:{order:np.zeros((2,n_pts_per_order,*data.shape[1::])) for order in orders} for par in range(input_dim)} for submeasure in submeasures}
    raw_measures = {measure:{ order: np.zeros( (2,n_pts_per_order) + data.shape[1::]) for order in orders} for measure in measures}

    if ref_values is None:
        #Calculate reference values
        orthoPoly = cp.orth_ttr(max_order, joint_dist)
        expansion_polynomial = cp.fit_regression(orthoPoly, samples.transpose(), data)
        ref_values = calculate_uqsa_measures(joint_dist, expansion_polynomial)

        # Monte Carlo Estimates
        ref_values["mean"] = np.mean(data, axis=0)
        ref_values["var"] = np.var(data, axis=0)

    for order in orders:
        dim = input_dim
        ncoefs = cp.bertran.terms(order,dim)
        sample_sizes = np.linspace(ncoefs, samples.shape[0], n_pts_per_order, dtype="int")
        orthoPoly = cp.orth_ttr(order, joint_dist)
        for idx, sample_size in enumerate(sample_sizes):
            expansion_polynomial = cp.fit_regression(orthoPoly, samples[0:sample_size].transpose(), 
                                                     data[0:sample_size])
            uqsa_data = calculate_uqsa_measures(joint_dist, expansion_polynomial)
            uhat = np.array([expansion_polynomial(*sample) for sample in samples])
            err = np.abs(uhat - data)
            rel_err = err/data
            if len(err.shape) ==1:
                err.shape = (err.shape[0],1)
                rel_err.shape = (rel_err.shape[0],1)
            
            
            abs_err_norm = np.array([np.linalg.norm(ui)/len(ui) for ui in err.T])
            rel_err_norm = np.array([np.linalg.norm(ui)/len(ui) for ui in rel_err.T])
            for uidx, err_val in enumerate(rel_err_norm):
                residual_error_dict[uidx][order][0] = sample_sizes #np.tile(sample_sizes, rel_err_norm.shape + (1,)).T 
                residual_error_dict[uidx][order][1,idx] = err_val
                residual_error_dict[uidx][order][2,idx] = abs_err_norm[uidx]

            for measure in measures:
                raw_measures[measure][order][0] = np.tile(sample_sizes, uqsa_data[measure].shape + (1,)).T 
                err = np.abs(uqsa_data[measure]-ref_values[measure])
                u = err/ref_values[measure] #TODO what if zero ref measure
                raw_measures[measure][order][1, idx] = u #data[measure]
                rel_err = np.linalg.norm(u.flat,ord=norm_ord)/len(u.flat) 
                abs_err = np.linalg.norm(err.flat,ord=norm_ord) #THIS DOESN'T MAKE ANY SENSE (averaging unnormalized absolute errors)
                # Store measures in dict position idx
                error_dict[measure][order][0] = sample_sizes
                error_dict[measure][order][1,idx] = rel_err
                error_dict[measure][order][2,idx] = abs_err

            for submeasure in submeasures:
                for par, value in suberror_dict[submeasure].items():
                    par_idx = int(par)
                    raw_sensitivities[submeasure][par][order][0] =  np.tile(sample_sizes, uqsa_data[submeasure][par_idx].shape + (1,)).T 
                    err = np.abs(uqsa_data[submeasure][par_idx] - ref_values[submeasure][par_idx])
                    u = err/(np.abs(ref_values[submeasure][par_idx])+zero_offset)  #TODO what if zero ref measure
                    raw_sensitivities[submeasure][par][order][1, idx] =  uqsa_data[submeasure][par_idx] 
                    rel_err = np.linalg.norm(u.flat,ord=norm_ord) 
                    abs_err = np.linalg.norm(err.flat,ord=norm_ord) 
                    # Store measures in dict position idx
                    value[order][0] = sample_sizes
                    value[order][1,idx] = rel_err
                    value[order][2,idx] = abs_err

    error_dict.update(suberror_dict)
    error_dict["res_err"] = residual_error_dict

    return error_dict, raw_measures, raw_sensitivities 

def plot_convergence_results(errors_dict, show_plots=False):
    max_ord = len(errors_dict["mean"])
    def plot_util_recursive(errors_dict,value_prefix='', fig_idx=None):
        for measure, value in errors_dict.items():
            if isinstance(next(iter(value.values())), dict): #TODO fix this hack
                plot_util_recursive(value, value_prefix=value_prefix + str(measure) +'-')
            else:
                if fig_idx is not None:
                    plt.figure(fig_idx)
                else:
                    plt.figure()
                    ax1 = plt.subplot(1,2,1)
                    ax2 = plt.subplot(1,2,2)

                for order,data in value.items():
                    if order < max_ord:
                        ax1.semilogy(data[0],(data[1]),label=order)
                        ax2.semilogy(data[0],(data[2]),label=order)
                        #ax1.plot(data[0],np.log10(data[1]),label=order)
                        #ax2.plot(data[0],np.log10(data[2]),label=order)
                    else:
                        ax1.semilogy(data[0,0:-1],(data[1,0:-1]),label=order)
                        ax2.semilogy(data[0,0:-1],(data[2,0:-1]),label=order)
                        #ax1.plot(data[0,0:-1],np.log10(data[1,0:-1]),label=order)
                        #ax2.plot(data[0,0:-1],np.log10(data[2,0:-1]),label=order)

                ax1.set_xlabel('n samples')
                ax1.set_ylabel('{} error'.format(value_prefix+str(measure)))
                ax2.set_xlabel('n samples')
                ax2.set_ylabel('{} error'.format(value_prefix+str(measure)))
                ax2.legend()

    plot_util_recursive(errors_dict)
    if show_plots: 
        plt.show()
