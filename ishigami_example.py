def ishigami_function(sample):
    q1 = sample[0]
    q2 = sample[1]
    q3 = sample[2]
    a = 7.
    b = 0.1
    return np.sin(q1) + a*np.sin(q2)**2 + b* q3**4 * np.sin(q1)

def ishigami_analytic():
    #Analytical values
    #Total variance
    measures = {}
    a = 7.
    measures["mean"] = a/2.0
    b = 0.1
    D = a**2./8 + b*np.pi**4./5 + b**2*np.pi**8./18 + 1./2
    measures 
    measures["var"] = D
    # Conditional variances
    D1 = b*np.pi**4./5 + b**2*np.pi**8./50. + 1./2
    D2 = a**2/8.
    D3 = 0
    
    D12  = 0
    D13  = b**2. * np.pi**8 / 18 - b**2*np.pi**8./50.
    D23  = 0
    D123 = 0
    
    # Main and total sensitivity indices
    measures["sens_m"] = {}
    measures["sens_m"][0] = D1/D
    measures["sens_m"][1] = D2/D
    measures["sens_m"][2] = D3/D
   

    measures["sens_t"] = {}
    measures["sens_t"][0] = (D1 + D12 + D13 + D123)/D
    measures["sens_t"][1] = (D2 + D12 + D23 + D123)/D
    measures["sens_t"][2] = (D3 + D13 + D23 + D123)/D
    return measures


