import numpy as np
import chaospy as cp
import multiprocessing
import matplotlib.pyplot as plt
import pickle
import h5py

from scipy import optimize


class ReymondModel(object):
    def __init__(self, distsMeta):
        self.distsMeta = distsMeta

    def __call__(self, samples):

        distsRanges = self.distsMeta['ranges']
        ## hayashi vs reymond area graphs
        mmHgToPa = 133.32
        PaTommHg = 1. / 133.32

        'c', 'As', 'Ps'
        P = np.linspace(75, 140, 100) * mmHgToPa
        rho = 1050.

        Ps = samples[0] * distsRanges['Ps'][1] + distsRanges['Ps'][0]
        As = samples[1] * distsRanges['As'][1] + distsRanges['As'][0]
        c = samples[2] * distsRanges['c'][1] + distsRanges['c'][0]

        # reymond
        'PmaxC', 'Pwidth', 'a', 'b'
        if len(samples) > 3:
            PmaxC = mmHgToPa * (samples[3] * distsRanges['PmaxC'][1] + distsRanges['PmaxC'][0])
            Pwidth = mmHgToPa * (samples[4] * distsRanges['Pwidth'][1] + distsRanges['Pwidth'][0])
            # a1      = samples[5]*distsRanges['a'][1] + distsRanges['a'][0]
            # b1      = samples[6]*distsRanges['b'][1] + distsRanges['b'][0]
        else:
            PmaxC = 20. * 133.32
            Pwidth = 30. * 133.32
        a1 = 0.4
        b1 = 5.0

        return self.calculateComplianceArea(P, rho, c, Ps, As, a1, b1, Pwidth, PmaxC)

    def calculateComplianceArea(self, P, rho, c, Ps, As, a1, b1, Pwidth, PmaxC):

        As0 = As  # 0.000665083
        Cs = As0 / (rho * c ** 2)  ##= 2.72562732294531E-008
        # print Cs, Cs*(a1 + b1/(1.0+(((100.*133.32)-PmaxC)/Pwidth)**2.0)),(a1 + b1/(1.0+(((100.*133.32)-PmaxC)/Pwidth)**2.0))
        a_reymond = Cs * (a1 * (P) + b1 * Pwidth * np.arctan(((P) - PmaxC) / Pwidth)) + As - Cs * (
        a1 * Ps + b1 * Pwidth * np.arctan((Ps - PmaxC) / Pwidth))
        C_reymond = Cs * (a1 + b1 / (1.0 + (((P) - PmaxC) / Pwidth) ** 2.0))

        # self.evaluationOfcoefficientsAB(Ps, PmaxC, Pwidth)

        return C_reymond, a_reymond

    def evaluationOfcoefficientsAB(self, Ps, PmaxC, Pwidth):
        '''
        Function to evaluate the coefficients a1 and b1 in the functional part
        '''

        a1 = 0.4
        b1 = 5.0

        x0 = [a1, b1]
        # print "Check C = Cs * artan(Ps); artan(Ps) =  ", (a1 + b1/(1.0+(((Ps)-PmaxC)/Pwidth)**2.0))


        func = lambda x, args: (x[0] + x[1] / (1.0 + (((args[0]) - args[1]) / args[2]) ** 2.0)) - 1.0

        args = [Ps, PmaxC, Pwidth]
        sol = optimize.root(func, x0, args=args)
        print "solution", sol.x


class HayashiModel(object):
    def __init__(self, distsMeta):
        self.distsMeta = distsMeta

    def __call__(self, samples):
        ## hayashi vs reymond area graphs
        mmHgToPa = 133.32
        PaTommHg = 1. / 133.32

        'c', 'As', 'Ps'
        P = np.linspace(75, 140, 100) * mmHgToPa
        rho = 1050.

        distsRanges = self.distsMeta['ranges']

        c = samples[0] * distsRanges['c'][1] + distsRanges['c'][0]
        As = samples[1] * distsRanges['As'][1] + distsRanges['As'][0]
        Ps = samples[2] * distsRanges['Ps'][1] + distsRanges['Ps'][0]

        return self.calculateComplianceArea(P, rho, c, Ps, As)

    def calculateComplianceArea(self, P, rho, c, Ps, As):
        # hayashi
        betaHayashi = 2. * rho * c ** 2. / Ps  ##= 3.663003663

        a_hayashi = As * (1.0 + np.log(P / Ps) / betaHayashi) ** 2.0
        C_hayashi = 2.0 * As / betaHayashi * (1.0 + np.log(P / Ps) / betaHayashi) / P

        return C_hayashi, a_hayashi


class LaplaceModel(object):
    def __init__(self, distsMeta):
        self.distsMeta = distsMeta

    def __call__(self, samples):
        ## hayashi vs reymond area graphs
        mmHgToPa = 133.32
        PaTommHg = 1. / 133.32

        'c', 'As', 'Ps'
        P = np.linspace(75, 140, 100) * mmHgToPa
        rho = 1050.

        distsRanges = self.distsMeta['ranges']

        c = samples[0] * distsRanges['c'][1] + distsRanges['c'][0]
        As = samples[1] * distsRanges['As'][1] + distsRanges['As'][0]
        Ps = samples[2] * distsRanges['Ps'][1] + distsRanges['Ps'][0]

        return self.calculateComplianceArea(P, rho, c, Ps, As)

    def calculateComplianceArea(self, P, rho, c, Ps, As):
        betaLaplace = 2 * rho * c ** 2 / np.sqrt(As) * As

        a_Laplace = ((P - Ps) * As / betaLaplace + np.sqrt(As)) ** 2.

        C_Laplace = (2. * ((P - Ps) * As / betaLaplace + np.sqrt(As))) * As / betaLaplace

        return C_Laplace, a_Laplace


def gpc(dists, distsMeta, wallModel, order, hdf5group, sampleScheme='M'):
    print "\n GeneralizedPolynomialChaos - order {}\n".format(order)

    dim = len(dists)

    expansionOrder = order
    numberOfSamples = 4 * cp.terms(expansionOrder, dim)

    # Sample in independent space
    samples = dists.sample(numberOfSamples, sampleScheme).transpose()
    model = wallModel(distsMeta)

    # Evaluate the model (which is not linear obviously)
    pool = multiprocessing.Pool()
    data = pool.map(model, samples)
    pool.close()
    pool.join()
    C_data = [retval[0] for retval in data]
    a_data = [retval[1] for retval in data]

    C_data = np.array(C_data)
    a_data = np.array(a_data)
    # Orthogonal C_polynomial from marginals
    orthoPoly = cp.orth_ttr(expansionOrder, dists)

    for data, outputName in zip([C_data, a_data], ['Compliance', 'Area']):

        # Fit the model together in independent space
        C_polynomial = cp.fit_regression(orthoPoly, samples.transpose(), data)

        # save data to dictionary
        plotMeanConfidenceAlpha = 5

        C_mean = cp.E(C_polynomial, dists)
        C_std = cp.Std(C_polynomial, dists)

        Si = cp.Sens_m(C_polynomial, dists)
        STi = cp.Sens_t(C_polynomial, dists)

        C_conf = cp.Perc(C_polynomial, [plotMeanConfidenceAlpha / 2., 100 - plotMeanConfidenceAlpha / 2.], dists)

        a = np.linspace(0, 100, 1000)
        da = a[1] - a[0]
        C_cdf = cp.Perc(C_polynomial, a, dists)

        C_pdf = da / (C_cdf[1::] - C_cdf[0:-1])
        # Resample to generate full histogram
        samples2 = dists.sample(numberOfSamples * 100, sampleScheme)
        C_data2 = C_polynomial(*samples2).transpose()

        # save in hdf5 file
        solutionDataGroup = hdf5group.create_group(outputName)

        solutionData = {'mean': C_mean,
                        'std': C_std,
                        'confInt': C_conf,
                        'Si': Si,
                        'STi': STi,
                        'cDataGPC': C_data,
                        'samplesGPC': samples,
                        'cData': C_data2,
                        'samples': samples2.transpose(),
                        'C_pdf': C_pdf}

        for variableName, variableValue in solutionData.iteritems():
            solutionDataGroup.create_dataset(variableName, data=variableValue)


#
# def mc_uq(dists,distsMeta, numberOfSamples, sampleMethod):
#
#     print "\n MonteCarlo - Ns: {}\n".format(numberOfSamples)
#     dim = len(dists)
#     # make sure numberOfSamples is even
#     # Sample in independent space
#     samples = dists.sample(numberOfSamples, sampleMethod).transpose()
#
#     model = modelOutputToQoI(distsMeta)
#     pool = multiprocessing.Pool()
#     print "evaluate samplesA"
#     data = pool.map(model,samples)
#     print "finished .."
#     pool.close()
#     pool.join()
#
#     C_data = [retval[0] for retval in data]
#
#     data = np.array(C_data)
#
#     ### statistics
#     # mean
#     mean = np.sum(data,axis=0)/numberOfSamples
#     variance = np.var(data,axis=0)
#
#     solutionData = {'mean':mean,
#                     'std': np.sqrt(variance),
#                     'cData':data,
#                     'samples':samples,
#                     }
#
#     ## Quantiles
#     plotMeanConfidenceAlpha = 5.
#     quantiles = [plotMeanConfidenceAlpha/2, 100.-plotMeanConfidenceAlpha/2]
#     solutionData['confInt'] = np.percentile(data,quantiles, axis=0)
#     return solutionData
#
# def mc(dists,distsMeta, numberOfSamples, sampleMethod):
#
#     print "\n MonteCarlo - Ns: {}\n".format(numberOfSamples)
#     dim = len(dists)
#     # make sure numberOfSamples is even
#     # Sample in independent space
#     samples = dists.sample(2.*numberOfSamples, sampleMethod).transpose()
#
#     samplesA = samples[0:numberOfSamples]
#     samplesB = samples[numberOfSamples::]#dists.sample(numberOfSamples, sampleMethod).transpose()
#     #samplesA = dists.sample(numberOfSamples, sampleMethod).transpose()
#     #samplesB = dists.sample(numberOfSamples, sampleMethod).transpose()
#
#     samplesTest = np.sum((samplesA-samplesB).ravel())
#     if samplesTest == 0:
#         print "WARNING: samplesA and samplesB are the same!"
#
#     samplesC = np.empty((dim,numberOfSamples,dim))
#     # create C sample matrices
#     for i in xrange(dim):
#         samplesC[i,:,:] = samplesB.copy()
#         samplesC[i,:,i] = samplesA[:,i].copy()
#
#     model = modelOutputToQoI(distsMeta)
#
#     pool = multiprocessing.Pool()
#     # dataA  = np.array(pool.map(model,samplesA))
#     # dataB  = np.array(pool.map(model,samplesB))
#     print "evaluate samplesA"
#     dataA = pool.map(model,samplesA)
#     print "finished .."
#     print "evaluate samplesB"
#     dataB = pool.map(model,samplesB)
#     print "finished .."
#     pool.close()
#     pool.join()
#
#     C_dataA = [retval[0] for retval in dataA]
#     C_dataB = [retval[0] for retval in dataB]
#
#     dataA = np.array(C_dataA)
#     dataB = np.array(C_dataB)
#     dataC  = np.empty((dim,numberOfSamples,1))
#     C_dataC = np.empty((dim,numberOfSamples,1))
#
#     pool = multiprocessing.Pool()
#     for i in xrange(dim):
#         print "evaluate samplesC ",i
#         temp_data  = pool.map(model,samplesC[i])
#         print "finished .."
#         C_dataC[i] = np.array([retval[0] for retval in temp_data])
#     pool.close()
#     pool.join()
#     dataC = C_dataC
#
#     ### statistics
#     # mean
#     meanA = np.sum(dataA,axis=0)/numberOfSamples
#     meanB = np.sum(dataB,axis=0)/numberOfSamples
#     #print np.shape(dataC), dataC
#     #meanC = np.mean(dataC,axis= 0)
#     #print np.shape(meanC), meanC
#     mean = np.mean([meanA,meanB])
#     dataAmm = dataA -mean
#     dataBmm = dataB -mean
#     dataCmm = dataC -mean
#     # sensitivity
#     f0sq = np.mean(dataAmm*dataBmm, axis=0)
#
#     varianceA = np.sum(dataAmm**2,axis=0)/numberOfSamples - f0sq
#     varianceB = np.sum(dataAmm**2,axis=0)/numberOfSamples - f0sq
#
#     # conditional variance
#     conditionalVarianceGivenQ = np.empty((dim,1))
#     conditionalVarianceNotQ   = np.empty((dim,1))
#
#     Si  =  []
#     STi =  []
#
#     for i in xrange(dim):
#         conditionalVarianceGivenQ[i] = np.sum(dataAmm*dataCmm[i],axis=0)/numberOfSamples - f0sq
#         conditionalVarianceNotQ[i]   = np.sum(dataBmm*dataCmm[i],axis=0)/numberOfSamples - f0sq
#         Si.append(conditionalVarianceGivenQ[i]/varianceA)
#         STi.append(1.- conditionalVarianceNotQ[i]/varianceA)
#
#     Si  = np.array(Si)
#     STi = np.array(STi)
#
#
# #     for i in xrange(dim):
# #         print "coefficients ", distsMeta['uncertainVariables'][i], '\n'
# #         print "     C_T    "
# #         print 'Si  ', '   '.join(['{:.3}'.format(j) for j in Si[i]])
# #         print "STi ", '   '.join(['{:.3}'.format(j) for j in STi[i]])
# #
#     solutionData = {'mean':meanA,
#                     'std': np.sqrt(varianceA),
#                     'Si':Si,
#                     'STi':STi,
#                     'cData':dataA,
#                     'dataA':dataA,
#                     'dataB':dataB,
#                     'dataC':dataC,
#                     'samples':samplesA,
#                     'samplesB':samplesB,
#                     'samplesC':samplesC
#                     }
#
#     ## Quantiles
#     plotMeanConfidenceAlpha = 5.
#     dataC = np.vstack([dataA,dataB])
#     quantiles = [plotMeanConfidenceAlpha/2, 100.-plotMeanConfidenceAlpha/2]
#     solutionData['confInt'] = np.percentile(dataC,quantiles, axis=0)
#
#
#     return solutionData
#
#
#
# def printOutSiSTiLatex(distsMeta,Si,STi):
#
#     ## latex table
#     emptyLine = ''.join(['                   &',
#                          "& ".join(['{:<10}'.format('') for j in distsMeta['uncertainVariables']]),
#                          '\\\\'])
#     parameter = ''.join(['                   &',
#                          "& ".join(['{:<10}'.format(j) for j in distsMeta['uncertainVariables']]),
#                          '\\\\'])
#     C_T = ''.join(['Compliance &',
#                          "& ".join(['{:<10}'.format('') for j in distsMeta['uncertainVariables']]),
#                          '\\\\'])
#
#
#     print C_T
#     print "\midrule"
#     print emptyLine
#     print parameter
#     index = 0
#     print  ''.join(['Si &',
#                     "& ".join(['{:<10.3}'.format(j) for j in Si[:,index]]),
#                     '\\\\'])
#     print  ''.join(['STi &',
#                     "& ".join(['{:<10.3}'.format(j) for j in STi[:,index]]),
#                     '\\\\'])
#     print emptyLine


## PLOTTING
colors = [{'hist_bar': '#0072bd',
           'Si': '#0072bd',
           'STi': '#d95319'},
          {'hist_bar': '#0072bd',
           'Si': '#4dbeee',
           'STi': '#edb120'}
          ]
xy_label_size = 16
annotation_size = 14
ymax = 0


def fixAxesMarks():
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='x', top='off')
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', right='off')


def plotHistogramPDF(sol):
    n, bins, patches = plt.hist(sol['cData'], bins=200, normed=True, histtype='stepfilled')  # ,color='#2b83ba')
    plt.setp(patches, 'edgecolor', plt.getp(patches[0], 'facecolor'))
    C_mean = sol['mean']
    C_std = sol['std']
    plt.xlabel(r'$C_T$', size=xy_label_size)
    plt.ylabel(r'$\rho(C_T)$ [' + sol['case'] + ']', size=xy_label_size)

    ylim = plt.gca().get_ylim()
    ann_pos = ylim[1]
    ylim = (ylim[0], ylim[1] * 1.1)
    global ymax
    ymax = max(ymax, ylim[1])
    ylim = (ylim[0], ymax)
    plt.gca().set_ylim(ylim)

    ax = plt.gca()
    ax.set_ylim([0, 1.7])
    ax.set_yticks([0, 0.4, 0.8, 1.2, 1.6])
    ax.set_yticklabels([0, 0.4, 0.8, 1.2, 1.6])  # ],['0','0.4','0.8','1.2','1.6'])

    plt.axvline(x=C_mean, linewidth=3, color='k')
    plt.annotate(r'$\mu$', (C_mean * 1.05, ann_pos), size=annotation_size)
    plt.axvline(x=C_mean + C_std, linewidth=2, linestyle='--', color='k')
    plt.annotate(r'$\mu+\sigma$', (C_mean + C_std * 1.05, ann_pos), size=annotation_size)
    plt.axvline(x=C_mean - C_std, linewidth=2, linestyle='--', color='k')
    plt.annotate(r'$\mu-\sigma$', (C_mean - C_std, ann_pos), size=annotation_size)
    plt.axvline(x=sol['confInt'][0], linewidth=2, linestyle='-.', color='k')
    plt.annotate(r'$y_{[1-\beta / 2]}$', (sol['confInt'][0], ann_pos), size=annotation_size)
    plt.axvline(x=sol['confInt'][1], linewidth=2, linestyle='-.', color='k')
    plt.annotate(r'$y_{[\beta / 2]}$', (sol['confInt'][1], ann_pos), size=annotation_size)
    plt.gca().set_xlim([0.5, 3.0])

    fixAxesMarks()
    # plt.tight_layout()


def plotSobolBarPlotVert(sol, distsMeta, idx=0, n_compare=1):
    Si = sol['Si']
    STi = sol['STi']
    index = np.arange(len(Si))

    phi = 1.61
    w_scale = 2.
    index = w_scale * index
    bar_width = w_scale / (2 * n_compare + (n_compare - 1) / phi + phi)

    ax = plt.gca()
    fixAxesMarks()

    left = index + idx * (2 + 1. / phi) * bar_width

    colorSi = '0.9'  # colors[idx]['Si']
    colorSTi = '0.2'  # colors[idx]['STi']

    rects = ax.bar(left, Si, bar_width, color=colorSi, label=r'$S_i$')

    ax.bar(left + bar_width, STi, bar_width, color=colorSTi, label=r'$S_{Ti}$')

    ax.set_ylim([-0.1, 1.])
    xlim = ax.get_xlim()
    xlim = (xlim[0], 4 * w_scale)
    ax.set_xlim(xlim)
    ax.set_ylabel('sensitivity')
    plt.xticks(index + n_compare * bar_width, (distsMeta['uncertainVariables']))
    ax.legend(loc='upper left')
    fixAxesMarks()
    plt.tight_layout()


def plotSobolBarPlotHorz(solutionData, distsMeta, idx=0, n_compare=1):
    phi = 1.61
    w_scale = 2.
    bar_width = w_scale / (2 * n_compare + (n_compare - 1) / phi + phi)
    rectsB = []
    labelsB = []
    max_length = 0

    for idx, sol in enumerate(solutionData):
        Si = sol['Si']
        STi = sol['STi']

        print max_length

        max_length = np.amax([max_length, np.amax([Si, STi])])
        index = np.arange(len(Si))
        index = w_scale * index

        ax = plt.gca()
        fixAxesMarks()

        left = index + idx * (2 + 1. / phi) * bar_width
        rectsB.append(ax.barh(left, Si, bar_width, color=colors[idx]['Si'], label=sol['case'] + r'$S_i$'))
        labelsB.append([sol['case'] + r'$S_i$' for S in Si])
        rectsB.append(
            ax.barh(left + bar_width, STi, bar_width, color=colors[idx]['STi'], label=sol['case'] + r'$S_{Ti}$'))
        labelsB.append([sol['case'] + r'$S_{Ti}$' for S in STi])

    for rects, labels in zip(rectsB, labelsB):
        for rect, label in zip(rects, labels):
            print rect
            width = rect.get_width()
            if (width / max_length) > 0.25:
                plt.text(0.95 * width, rect.get_y(), '%s' % (label), ha='right', va='bottom')
            else:
                plt.text(1.05 * width, rect.get_y(), '%s' % (label), ha='left', va='bottom')

    ax.set_xlim([0, 1])
    ylim = ax.get_ylim()
    ylim = (ylim[0], 4 * w_scale)
    ax.set_ylim(ylim)
    ax.set_xlabel('sensitivity')
    plt.yticks(index + n_compare * bar_width, (distsMeta['uncertainVariables']))
    fixAxesMarks()
    plt.tight_layout()


def plot(solutionData, distsMeta):
    histCmpFig = plt.figure()
    SobolIndicesFigVert = plt.figure()
    # SobolIndicesFigHorz = plt.figure()
    # plt.figure(SobolIndicesFigHorz.number)
    # plotSobolBarPlotHorz(solutionData, distsMeta, idx=None,n_compare=2)
    plt.savefig('WK2SensitivitiesHorz.pdf')

    for idx, sol in enumerate(solutionData):
        print "start with ", sol['case']
        print 'mean', sol['mean']
        print 'std', sol['std']
        print 'confInt', sol['confInt']

        #         plt.figure()
        #         plotHistogramPDF(sol)
        #         plt.savefig(sol['case']+'WK2Histogram.pdf')
        #
        # Plot Histograms on top of each other
        plt.figure(histCmpFig.number)
        plt.subplot(2, 1, idx)

        plotHistogramPDF(sol)
        plt.savefig('WK2Histogram.eps')

        # Barplot of Sensitivities
        ## Si STi
        plt.figure(SobolIndicesFigVert.number)
        plotSobolBarPlotVert(sol, distsMeta, idx=idx, n_compare=2)
        plt.savefig('WK2SensitivitiesVert.pdf')
        print 'Si', sol['Si']
        print 'STi', sol['STi']

        #         plt.figure()
        #         plotSobolBarPlot(sol,distsMeta)
        #         plt.savefig(sol['case']+'WK2Sensitivities.pdf')

        plt.figure()
        dim = len(distsMeta['uncertainVariables'])
        for index, uncertainVariable in enumerate(distsMeta['uncertainVariables']):

            if index < 2.:
                ax1 = plt.subplot(2, 2, index + 1)
            else:
                ax1 = plt.subplot(2, 2, index + 1)

            # ax1.plot(sol['samples2'][:,index],sol['cData2'],marker = '+', linestyle = 'g')
            print sol['case']
            print np.shape(sol['samples'][:, index]), np.shape(sol['cData'].ravel())
            print

            samples = sol['samples'][:, index]
            distsRanges = distsMeta['ranges']
            if index == 0:
                samples = samples * distsRanges['H'][1] + distsRanges['H'][0]
            elif index == 1:
                samples = (samples * distsRanges['SV'][1] + distsRanges['SV'][0])  # *1e-6
            elif index == 2:
                samples = (samples * distsRanges['Pmax'][1] + distsRanges['Pmax'][
                    0])  # *133.32 #*3.52355288029e-08#*11.6*1.e-6*1e-3
            elif index == 3:
                samples = (samples * distsRanges['Pmin'][1] + distsRanges['Pmin'][0])  # *133.32

            ax1.scatter(samples, sol['cData'], marker='+', facecolor='', lw=0.0, edgecolor='k', s=4)

            plt.legend()

            ax1.set_xlabel(uncertainVariable)
            ax1.set_ylabel(r'$C_T$')

            ax1.spines['top'].set_visible(False)
            ax1.tick_params(axis='x', top='off')
            ax1.spines['right'].set_visible(False)
            ax1.tick_params(axis='y', right='off')

        plt.tight_layout()
        plt.savefig(sol['case'] + 'WK2ScatterPlots.pdf')

        # plt.show()


def plotUQ(solutionData, distsMeta):
    histCmpFig = plt.figure()

    for idx, sol in enumerate(solutionData):

        print "start with ", sol['case']
        print 'mean', sol['mean']
        print 'std', sol['std']
        print 'confInt', sol['confInt']

        # Plot Histograms on top of each other
        plt.figure(histCmpFig.number)
        plt.subplot(2, 1, idx)

        plotHistogramPDF(sol)
        ylim = plt.gca().get_ylim()
        global ymax
        ymax = max(ymax, ylim[1])
        ylim = (ylim[0], ymax)
        plt.gca().set_ylim(ylim)
        ax = plt.gca()
        ax.set_yticks(np.linspace(0, 2, 5))
        if idx > 0:
            plt.subplot(2, 1, 1)
            ax = plt.gca()
            ax.set_ylim(ylim)
            ax.set_yticks(np.linspace(0, 2, 5))
            plt.savefig('WK2MRIHistogram.pdf')
            plt.savefig('WK2MRIHistogram.eps')


def plotUQCmp(solutionData, distsMeta, fname='WK2CmpHistogram'):
    gpcFig = plt.figure()  # (figsize=(16,9), dpi=800,edgecolor='k')
    mcFig = plt.figure(figsize=(6, 3), dpi=800, edgecolor='k')
    for idx, sols in enumerate(solutionData):

        sol = sols[0]
        print "start with ", sol['case']
        print 'mean', sol['mean']
        print 'std', sol['std']
        print 'confInt', sol['confInt']
        # Plot Histograms on top of each other
        plt.figure(gpcFig.number)
        initialBins = np.linspace(0.5, 3.0, 200)
        n, bins, patches = plt.hist(sol['cData'], bins=initialBins, normed=True, alpha=1 - idx * 0.25,
                                    histtype='stepfilled')  # ,color='#2b83ba')
        plt.setp(patches, 'edgecolor', plt.getp(patches[0], 'facecolor'))
        plt.xlabel(r'$C_T$', size=xy_label_size)
        plt.ylabel('Density [' + sol['case'] + ']', size=xy_label_size)
        ax = plt.gca()
        ax.set_xlim([0.5, 3.0])
        ax.set_ylim([0.0, 9.0])  # to get nice aspect ration :)
        if idx > 0:
            fixAxesMarks()
            plt.savefig(fname + 'gpc.pdf')

        sol = sols[1]
        print "start with ", sol['case']
        print 'mean', sol['mean']
        print 'std', sol['std']
        print 'confInt', sol['confInt']
        # Plot Histograms on top of each other
        plt.figure(mcFig.number)
        n, bins, patches = plt.hist(sol['cData'], bins=initialBins, normed=True, alpha=1 - idx * 0.25,
                                    histtype='stepfilled')  # ,color='#2b83ba')
        plt.setp(patches, 'edgecolor', plt.getp(patches[0], 'facecolor'))
        plt.xlabel(r'$C_T$', size=xy_label_size)
        plt.ylabel('Density [' + sol['case'] + ']', size=xy_label_size)
        ax = plt.gca()
        ax.set_xlim([0.5, 3.0])
        if idx > 0:
            fixAxesMarks()
            plt.savefig(fname + 'mc.pdf')


def distributions(model='Hayashi', devPar=0.05):
    rho = 1050.
    mmHgToPa = 133.32
    cm2Tom2 = 1. / 100. / 100.

    dev = 0.10

    As = 5.12 * cm2Tom2
    AsA = np.array([As * (1. - dev), As * (1. + dev) - As * (1. - dev)])
    AsRV = cp.Uniform(0, 1)

    c = 6.25609258389  # np.sqrt(As/(Cs *rho))
    cA = np.array([c * (1. - dev), c * (1. + dev) - c * (1. - dev)])
    cRV = cp.Uniform(0, 1)

    Ps = 100 * mmHgToPa
    PsA = np.array([Ps * (1. - dev), Ps * (1. + dev) - Ps * (1. - dev)])
    PsRV = cp.Uniform(0, 1)

    distsMeta = {'uncertainVariables': ['Ps', 'As', 'c']}
    distsRange = {'c': cA, 'As': AsA, 'Ps': PsA}

    if model == "Reymond":

        dev = devPar

        # reymond additional parameters

        PmaxC = 20.
        PmaxCA = np.array([PmaxC * (1. - dev), PmaxC * (1. + dev) - PmaxC * (1. - dev)])
        PmaxCRV = cp.Uniform(0, 1)

        Pwidth = 30.
        PwidthA = np.array([Pwidth * (1. - dev), Pwidth * (1. + dev) - Pwidth * (1. - dev)])
        PwidthRV = cp.Uniform(0, 1)

        #         a = 0.4
        #         aA = np.array([a*(1.-dev),a*(1.+dev)-a*(1.-dev)])
        #         aRV = cp.Uniform(0,1)
        #
        #         b =  5.0
        #         bA = np.array([b*(1.-dev),b*(1.+dev)-b*(1.-dev)])
        #         bRV = cp.Uniform(0,1)

        distsMeta['uncertainVariables'].extend(['PmaxC', 'Pwidth', 'a', 'b'])
        distsRange['PmaxC'] = PmaxCA
        distsRange['Pwidth'] = PwidthA
        #         distsRange['a'] = aA
        #         distsRange['b'] = bA
        #
        # dists = cp.J(cRV,AsRV,PsRV,PmaxCRV,PwidthRV,aRV,aRV)

        dists = cp.J(cRV, AsRV, PsRV, PmaxCRV, PwidthRV)  # ,aRV,bRV)

    else:
        dists = cp.J(cRV, AsRV, PsRV)

    distsMeta['ranges'] = distsRange

    return dists, distsMeta


def deterministicWallModelsQuantitaticePlot():
    '''
    comparision of the determinsitic wall models
    '''
    fullRange = True
    langewouter = True
    hayashi = True

    mmHgToPa = 133.32
    cm2Tom2 = 1. / 100. / 100.
    cm2Tomm2 = 10. * 10.
    m2Tomm2 = 1000 * 1000.
    m2Tocm2 = 100 * 100.
    mm2Tom2 = 1. / 1000. / 1000.
    mm2Tocm2 = 1. / 10. / 10.

    PaTommHg = 1. / 133.32

    if fullRange == True:
        #     ## FULL RANGE
        P = np.linspace(10, 200, 100) * mmHgToPa
    else:
        P = np.linspace(75, 143, 100) * mmHgToPa

    rho = 1050.

    # aortic arch II of 37 vessel 10 Mathys et al 2007
    As0 = 465.6802425 / m2Tomm2
    As = 0.0007173203
    Ps = 13089.9624419915  # *mmHgToPa#79.65*mmHgToPa # pressure mean
    c = 6.4796505546  # estimated carotid -> femoral wave speed

    PmaxC = 50. * mmHgToPa
    Pwidth = 30. * mmHgToPa
    a1 = 0.4  # 0.38372
    b1 = 5.0

    hayashiModel = HayashiModel(None)

    laplaceModel = LaplaceModel(None)

    # laplaceModel = LaplaceModel(None)
    # C_l0,A_l0 = laplaceModel.calculateComplianceArea(P,rho,c,0,As0)

    reymondModel = ReymondModel(None)

    # langewouter pressure area compliance, individ 2

    As = 5.12 * cm2Tom2
    Ps = 100 * mmHgToPa
    Cs = 16.61 * cm2Tom2 / mmHgToPa * 1.e-3
    c = np.sqrt(As / (Cs * rho))
    cEmp = (13.3 / (np.sqrt(As * m2Tomm2 * 4. / np.pi) ** 0.3))
    # c = cEmp
    # c = 5.5
    print "langwouter wavespeed", np.sqrt(As / (Cs * rho)), Cs, c, cEmp

    LangC_h, LangA_h = hayashiModel.calculateComplianceArea(P, rho, 5, Ps, As)
    LangC_l, LangA_l = laplaceModel.calculateComplianceArea(P, rho, 4, Ps, As)
    LangC_r, LangA_r = reymondModel.calculateComplianceArea(P, rho, c, Ps, As, a1, b1, Pwidth, PmaxC)

    fig = plt.figure(1, (8, 3), dpi=80)

    plt.plot(P * PaTommHg, LangA_h * m2Tocm2, linewidth=2, color='b')
    # plt.ylim([0,0.025])
    # plt.ylim([1,1.6])
    # plt.xlim([58,82])
    ax = fig.gca()
    ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks([])
    ax.xaxis.set_ticks([])
    # ax.set_ylabel('\rho(Z1)')
    # ax.tick_params(axis='y',left='off',right='off')
    ax.tick_params(axis='x', top='off')
    ax.tick_params(axis='y', right='off')

    fig = plt.figure(2, (8, 3), dpi=80)

    plt.plot(P * PaTommHg, LangA_l * m2Tocm2, linewidth=2, color='r')
    # plt.ylim([0,0.025])
    # plt.ylim([1,1.6])
    # plt.xlim([58,82])
    ax = fig.gca()
    ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks([])
    ax.xaxis.set_ticks([])
    # ax.set_ylabel('\rho(Z1)')
    # ax.tick_params(axis='y',left='off',right='off')
    ax.tick_params(axis='x', top='off')
    ax.tick_params(axis='y', right='off')

    fig = plt.figure(3, (8, 3), dpi=80)
    plt.plot(P * PaTommHg, LangA_r * m2Tocm2, linewidth=2, color='g')
    # plt.ylim([0,0.025])
    # plt.ylim([1,1.6])
    # plt.xlim([58,82])
    ax = fig.gca()
    ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks([])
    ax.xaxis.set_ticks([])
    # ax.set_ylabel('\rho(Z1)')
    # ax.tick_params(axis='y',left='off',right='off')
    ax.tick_params(axis='x', top='off')
    ax.tick_params(axis='y', right='off')

    fig = plt.figure(4, (8, 3), dpi=80)
    plt.plot(P * PaTommHg, LangA_r * m2Tocm2, linewidth=2, color='g')
    plt.plot(P * PaTommHg, LangA_l * m2Tocm2, linewidth=2, color='r')
    plt.plot(P * PaTommHg, LangA_h * m2Tocm2, linewidth=2, color='b')
    # plt.ylim([0,0.025])
    # plt.ylim([1,1.6])
    # plt.xlim([58,82])
    ax = fig.gca()
    ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks([])
    ax.xaxis.set_ticks([])
    # ax.set_ylabel('\rho(Z1)')
    # ax.tick_params(axis='y',left='off',right='off')
    ax.tick_params(axis='x', top='off')
    ax.tick_params(axis='y', right='off')

    plt.show()


def deterministicWallModels():
    '''
    comparision of the determinsitic wall models
    '''
    fullRange = True
    langewouter = True
    hayashi = True

    mmHgToPa = 133.32
    cm2Tom2 = 1. / 100. / 100.
    cm2Tomm2 = 10. * 10.
    m2Tomm2 = 1000 * 1000.
    m2Tocm2 = 100 * 100.
    mm2Tom2 = 1. / 1000. / 1000.
    mm2Tocm2 = 1. / 10. / 10.

    PaTommHg = 1. / 133.32

    if fullRange == True:
        #     ## FULL RANGE
        P = np.linspace(50, 200, 100) * mmHgToPa
    else:
        P = np.linspace(75, 143, 100) * mmHgToPa

    rho = 1050.

    # aortic arch II of 37 vessel 10 Mathys et al 2007
    As0 = 465.6802425 / m2Tomm2
    As = 0.0007173203
    Ps = 13089.9624419915  # *mmHgToPa#79.65*mmHgToPa # pressure mean
    c = 6.4796505546  # estimated carotid -> femoral wave speed

    PmaxC = 20. * mmHgToPa
    Pwidth = 30. * mmHgToPa
    a1 = 0.4  # 0.38372
    b1 = 5.0

    hayashiModel = HayashiModel(None)

    laplaceModel = LaplaceModel(None)

    # laplaceModel = LaplaceModel(None)
    # C_l0,A_l0 = laplaceModel.calculateComplianceArea(P,rho,c,0,As0)

    reymondModel = ReymondModel(None)

    # langewouter pressure area compliance, individ 2

    As = 5.12 * cm2Tom2
    Ps = 100 * mmHgToPa
    Cs = 16.61 * cm2Tom2 / mmHgToPa * 1.e-3
    c = np.sqrt(As / (Cs * rho))
    cEmp = (13.3 / (np.sqrt(As * m2Tomm2 * 4. / np.pi) ** 0.3))
    # c = cEmp
    # c = 5.5
    print "langwouter wavespeed", np.sqrt(As / (Cs * rho)), Cs, c, cEmp

    LangC_h, LangA_h = hayashiModel.calculateComplianceArea(P, rho, c, Ps, As)
    LangC_l, LangA_l = laplaceModel.calculateComplianceArea(P, rho, c, Ps, As)
    LangC_r, LangA_r = reymondModel.calculateComplianceArea(P, rho, c, Ps, As, a1, b1, Pwidth, PmaxC)

    import dataHayashi1947
    As = dataHayashi1947.abAs * mm2Tom2
    Ps = dataHayashi1947.Ps * mmHgToPa
    Cs = dataHayashi1947.abCs / mmHgToPa * mm2Tom2
    c = np.sqrt(As / (Cs * rho))
    cEmp = (13.3 / (np.sqrt(As * m2Tomm2 * 4. / np.pi) ** 0.3))
    # c = cEmp
    # c  = 8
    print "hayashi wavespeed abdominal", np.sqrt(As / (Cs * rho)), Cs, c, cEmp

    HayAbC_h, HayAbA_h = hayashiModel.calculateComplianceArea(P, rho, c, Ps, As)
    HayAbC_l, HayAbA_l = laplaceModel.calculateComplianceArea(P, rho, c, Ps, As)
    HayAbC_r, HayAbA_r = reymondModel.calculateComplianceArea(P, rho, c, Ps, As, a1, b1, Pwidth, PmaxC)

    As = dataHayashi1947.caAs * mm2Tom2
    Ps = dataHayashi1947.Ps * mmHgToPa
    Cs = dataHayashi1947.caCs / mmHgToPa * mm2Tom2
    c = np.sqrt(As / (Cs * rho))
    cEmp = (13.3 / (np.sqrt(As * m2Tomm2 * 4. / np.pi) ** 0.3))

    # c = cEmp
    # c  = 8
    print "hayashi wavespeed carotid", np.sqrt(As / (Cs * rho)), Cs, c, cEmp

    HayCaC_h, HayCaA_h = hayashiModel.calculateComplianceArea(P, rho, c, Ps, As)
    HayCaC_l, HayCaA_l = laplaceModel.calculateComplianceArea(P, rho, c, Ps, As)
    HayCaC_r, HayCaA_r = reymondModel.calculateComplianceArea(P, rho, c, Ps, As, a1, b1, Pwidth, PmaxC)

    fig, (ax, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(8.27 / 1.5, 11.69 / 3.), dpi=100)

    plt.rc('text', usetex=True)
    plt.subplots_adjust(right=0.96)
    plt.subplots_adjust(left=0.15)
    plt.subplots_adjust(top=0.90)
    plt.subplots_adjust(bottom=0.17)
    plt.subplots_adjust(hspace=0.29)
    plt.subplots_adjust(wspace=0.34)

    # ax.plot(P*PaTommHg, A_l0*1000*1000,'m',label='Laplace 0')

    # area langwouters pressure in mmHg area in cm2
    langewouterPA = np.array([19.780043991201765, 2.0512393521295733,
                              40.179392692890005, 2.9862411517696454,
                              59.86688376610394, 3.934999400119975,
                              79.3612706030223, 4.7068526294741035,
                              99.35955666009657, 5.111102979404118,
                              119.52352386665528, 5.36559728054389,
                              140.03085097266268, 5.473493701259747,
                              159.82403519296145, 5.547924815036991,
                              179.97086297026317, 5.598255948810236
                              ]).reshape((9, 2)).T

    if langewouter:
        error = langewouterPA[1] * np.linspace(0.03, 0.07, len(langewouterPA[1]))
        ax.errorbar(langewouterPA[0], langewouterPA[1] * cm2Tomm2, yerr=error * cm2Tomm2, color='k',
                    label='Langewouter', marker="o", linestyle='')
        ax.plot(P * PaTommHg, LangA_r * m2Tomm2, 'g-', label='reymond', alpha=0.6)
        ax.plot(P * PaTommHg, LangA_h * m2Tomm2, 'b:', label='hayashi')
        ax.plot(P * PaTommHg, LangA_l * m2Tomm2, 'r--', label='Laplace')

    if hayashi:
        ax2.errorbar(dataHayashi1947.HayashiAbdominalAortaAreaPressure[1],
                     dataHayashi1947.HayashiAbdominalAortaAreaPressure[0],
                     yerr=dataHayashi1947.abareaError, color='k', marker="o", linestyle='')

        ax2.plot(P * PaTommHg, HayAbA_r * m2Tomm2, 'g-', label='reymond', alpha=0.6)
        ax2.plot(P * PaTommHg, HayAbA_h * m2Tomm2, 'b:', label='hayashi')
        ax2.plot(P * PaTommHg, HayAbA_l * m2Tomm2, 'r--', label='Laplace')

        ax3.errorbar(dataHayashi1947.HayashiCarotidAortaAreaPressure[1],
                     dataHayashi1947.HayashiCarotidAortaAreaPressure[0],
                     yerr=dataHayashi1947.caareaError, color='k', marker="o", linestyle='')

        ax3.plot(P * PaTommHg, HayCaA_r * m2Tomm2, 'g-', label='reymond', alpha=0.6)
        ax3.plot(P * PaTommHg, HayCaA_h * m2Tomm2, 'b:', label='hayashi')
        ax3.plot(P * PaTommHg, HayCaA_l * m2Tomm2, 'r--', label='Laplace')

    # zoom-in / limit the view to different portions of the data

    if fullRange == True:

        #     ## FULL RANGE
        ax.set_ylim([300, 700])  # outliers only
        ax2.set_ylim([38, 64])
        ax3.set_ylim([10.5, 18.2])  # most of the data
        ax.set_xlim([45, 205])

    else:
        ax.set_ylim([400, 650])  # outliers only
        ax2.set_ylim([45, 65])
        ax3.set_ylim([10.5, 18.2])  # most of the data
        ax.set_xlim([72, 145])
    # hide the spines between ax and ax2
    ax.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)

    ax3.spines['top'].set_visible(False)
    # ax2.spines['bottom'].set_visible(False)

    ax.xaxis.tick_top()
    ax.tick_params(labeltop='off')  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()
    ax2.tick_params(labeltop='off')
    ax2.tick_params(labelbottom='off')
    ax2.tick_params(axis='x', bottom='off')

    # This looks pretty good, and was fairly painless, but you can get that
    # cut-out diagonal lines look with just a bit more work. The important
    # thing to know here is that in axes coordinates, which are always
    # between 0-1, spine endpoints are at these locations (0,0), (0,1),
    # (1,0), and (1,1).  Thus, we just need to put the diagonals in the
    # appropriate corners of each of our axes, and so long as we use the
    # right transform and disable clipping.

    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal
    # ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (-d, +d), **kwargs)
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    # ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    kwargs.update(transform=ax3.transAxes)  # switch to the bottom axes
    ax3.plot((-d, +d), (1 - d, 1 + d), **kwargs)

    ax2.set_ylabel(r'Area $[mm^2]$')
    ax3.set_xlabel(r'Pressure $[mmHg]$')

    #     # old data
    #     plotDataLang = np.array([658.1959466554663, 13.838065735587527,
    #     4342.329097973328, 16.320224719101127,
    #     7018.37656200777, 16.287997479785783,
    #     9605.796492701882, 15.391588785046732,
    #     12210.227869368897, 13.274797857817916,
    #     14765.725086632367, 10.446266932689277,
    #     17371.416570408484, 8.405743988239,
    #     19947.915572823695, 6.8483461094193,
    #     22536.17557492387, 6.002782736532605]).reshape((9,2)).T
    #
    #     plotDataLangPressure = plotDataLang[0]/133.32
    #     plotDataLangCompliance = plotDataLang[1]*1.e-3

    # compliance langewouter in pressure mmHg, compliance cm2/mmHg
    #     langewouterPC = np.array([ #30.97440680781466, 50.049320204977604,
    #                                # 51.10524769047947, 48.6755350791006,
    #                                 71.14792430650493, 39.22151802166829,
    #                                 90.67375388114431, 22.600938368739705,
    #                                 111.19813960619962, 10.966227287491071,
    #                                 130.74313532576053, 6.582940701289387,
    #                                 150.70148090413096, 4.430847936717882,
    #                                # 170.7441575201563, 3.5702654715318927
    #                                ]).reshape((5,2)).T
    #
    #     compCLangewouter = (langewouterPA[1][1::]-langewouterPA[1][0:-1])/(langewouterPA[0][1::]-langewouterPA[0][0:-1])
    #     compC_P = (langewouterPA[0][1::]+langewouterPA[0][0:-1])/2.0
    #
    #     # error plot
    #     #ax.plot(P*PaTommHg, (a_reymond - a_hayashi)**2./ (a_hayashi)**2,'g',label='hayashi')
    #     #ax.plot([50,180],[1.e-7,1.e-7],'k')
    #     ax2 = plt.subplot(1,2,2)
    #
    #     ax2.plot(P*PaTommHg, LangC_r*m2Tocm2/PaTommHg/1.e-3,'g',label='reymond')
    #     ax2.plot(P*PaTommHg, LangC_h*m2Tocm2/PaTommHg/1.e-3,'b',label='hayashi')
    #     ax2.plot(P*PaTommHg, LangC_l*m2Tocm2/PaTommHg/1.e-3,'r',label='Laplace')
    #     #ax2.plot(P*PaTommHg, C_l0*1000/PaTommHg,'m',label='Laplace 0')
    #     #ax2.plot(plotDataLangPressure,plotDataLangCompliance,'k')
    #
    # 'langewouters'
    Pla = langewouterPA[0]  # [3:7]
    Ala = langewouterPA[1]  # [3:7]
    Alr = np.interp(Pla, P * PaTommHg, LangA_r * m2Tocm2)
    Alh = np.interp(Pla, P * PaTommHg, LangA_h * m2Tocm2)
    All = np.interp(Pla, P * PaTommHg, LangA_l * m2Tocm2)
    L_rmsA_r = np.mean(np.sqrt(((Alr - Ala) / Ala) ** 2.))
    L_rmsA_h = np.mean(np.sqrt(((Alh - Ala) / Ala) ** 2.))
    L_rmsA_l = np.mean(np.sqrt(((All - Ala) / Ala) ** 2.))
    L_R2_r = 1 - np.sum((Ala - Alr) ** 2.) / np.sum((Ala - np.mean(Ala)) ** 2.)
    L_R2_h = 1 - np.sum((Ala - Alh) ** 2.) / np.sum((Ala - np.mean(Ala)) ** 2.)
    L_R2_l = 1 - np.sum((Ala - All) ** 2.) / np.sum((Ala - np.mean(Ala)) ** 2.)

    # hayashi abdominal
    Pha = dataHayashi1947.HayashiAbdominalAortaAreaPressure[1]  # [8:15]
    Aha = dataHayashi1947.HayashiAbdominalAortaAreaPressure[0]  # [8:15]

    Alr = np.interp(Pha, P * PaTommHg, HayAbA_r * m2Tomm2)
    Alh = np.interp(Pha, P * PaTommHg, HayAbA_h * m2Tomm2)
    All = np.interp(Pha, P * PaTommHg, HayAbA_l * m2Tomm2)

    HA_R2_r = 1 - np.sum((Aha - Alr) ** 2.) / np.sum((Aha - np.mean(Aha)) ** 2.)
    HA_R2_h = 1 - np.sum((Aha - Alh) ** 2.) / np.sum((Aha - np.mean(Aha)) ** 2.)
    HA_R2_l = 1 - np.sum((Aha - All) ** 2.) / np.sum((Aha - np.mean(Aha)) ** 2.)

    HA_rmsA_r = np.mean(np.sqrt(((Alr - Aha) / Aha) ** 2.))
    HA_rmsA_h = np.mean(np.sqrt(((Alh - Aha) / Aha) ** 2.))
    HA_rmsA_l = np.mean(np.sqrt(((All - Aha) / Aha) ** 2.))
    # Hayashi carotid
    Pha = dataHayashi1947.HayashiCarotidAortaAreaPressure[1]  # [8:15]
    Aha = dataHayashi1947.HayashiCarotidAortaAreaPressure[0]  # [8:15]

    Alr = np.interp(Pha, P * PaTommHg, HayCaA_r * m2Tomm2)
    Alh = np.interp(Pha, P * PaTommHg, HayCaA_h * m2Tomm2)
    All = np.interp(Pha, P * PaTommHg, HayCaA_l * m2Tomm2)

    HC_R2_r = 1 - np.sum((Aha - Alr) ** 2.) / np.sum((Aha - np.mean(Aha)) ** 2.)
    HC_R2_h = 1 - np.sum((Aha - Alh) ** 2.) / np.sum((Aha - np.mean(Aha)) ** 2.)
    HC_R2_l = 1 - np.sum((Aha - All) ** 2.) / np.sum((Aha - np.mean(Aha)) ** 2.)

    HC_rmsA_r = np.mean(np.sqrt(((Alr - Aha) / Aha) ** 2.))
    HC_rmsA_h = np.mean(np.sqrt(((Alh - Aha) / Aha) ** 2.))
    HC_rmsA_l = np.mean(np.sqrt(((All - Aha) / Aha) ** 2.))

    # HC_rmsA_r = np.mean(np.sqrt( ((np.interp(Pha, P*PaTommHg, HayCaA_r*m2Tomm2)-Aha)/Aha)**2. ) )
    # HC_rmsA_h = np.mean(np.sqrt( ((np.interp(Pha, P*PaTommHg, HayCaA_h*m2Tomm2)-Aha)/Aha)**2. ) )
    # HC_rmsA_l = np.mean( np.sqrt(((np.interp(Pha, P*PaTommHg, HayCaA_l*m2Tomm2)-Aha)/Aha)**2. ) )



    print "RMS error area"
    print "data, laplace, hayashi, reymond"
    # print "Langewouter et al. 1985 thoracic aorta," ,L_rmsA_l,',',L_R2_l,',',L_rmsA_h,',',L_R2_h,',',L_rmsA_r,',',L_R2_r
    # print "Hayashi et al. 1974 abdominal aorta," ,HA_rmsA_l,',',HA_R2_l,',',HA_rmsA_h,',',HA_R2_h,',',HA_rmsA_r,',',HA_R2_r
    # print "Hayashi et al. 1974 carotid aorta," ,HC_rmsA_l,',',HC_R2_l,',',HC_rmsA_h,',',HC_R2_h,',',HC_rmsA_r,',',HC_R2_r

    print "Langewouter et al. 1985 thoracic aorta,", L_rmsA_l, ',', L_rmsA_h, ',', L_rmsA_r
    print "Hayashi et al. 1974 abdominal aorta,", HA_rmsA_l, ',', HA_rmsA_h, ',', HA_rmsA_r
    print "Hayashi et al. 1974 carotid aorta,", HC_rmsA_l, ',', HC_rmsA_h, ',', HC_rmsA_r

    # #
    # #
    # #     rmsC_r = np.mean(np.sqrt((np.interp(langewouterPC[0], P*PaTommHg, C_r*m2Tocm2/PaTommHg/1.e-3)-langewouterPC[1])**2.)/langewouterPC[1])
    # #     rmsC_h = np.mean(np.sqrt((np.interp(langewouterPC[0], P*PaTommHg, C_h*m2Tocm2/PaTommHg/1.e-3)-langewouterPC[1])**2.)/langewouterPC[1])
    # #     rmsC_l = np.mean(np.sqrt((np.interp(langewouterPC[0], P*PaTommHg ,C_l*m2Tocm2/PaTommHg/1.e-3)-langewouterPC[1])**2.)/langewouterPC[1])
    # #
    # #
    # #     print "RMS error compliance, laplace, hayashi, reymond",rmsC_l,rmsC_h,rmsC_r
    # #
    #     if langewouter:
    #         ax2.plot(langewouterPC[0], langewouterPC[1],'k',label='Langewouter',marker = "o", linestyle = '')
    #         ax2.plot(compC_P, compCLangewouter/1.e-3,'m',label='Langewouter',marker = "d", linestyle = '')
    #
    #     if hayashi:
    #         ax2.plot(dataHayashi1947.CPressure,dataHayashi1947.C*mm2Tocm2/1.e-3,label='estimate',marker = "o", linestyle = '')
    #
    #     #ax2.set_yscale("log")
    #
    #     #ax2.set_ylim([1,60])
    #
    #     plt.ylabel(r'Compliance $[10^{-3}cm^2/mmHg]$')
    #     plt.xlabel(r'Pressure $[mmHg]$')
    #
    #
    #     #
    #     # ax3 = plt.subplot(3,1,3)
    #     #
    #     # ax3.plot(P*PaTommHg, np.sqrt(a_reymond/(C_reymond*rho)),'b',label='reymond')
    #     # ax3.plot(P*PaTommHg, np.sqrt(a_hayashi/(C_hayashi*rho)),'r',label='hayashi')
    #     #
    #     # plt.ylabel('wavespeed [m/s]')
    #     # plt.xlabel('Pressure [mmHg]')


    if fullRange == True:
        ax.add_patch(plt.Rectangle((75, 0), 70, 700, facecolor="yellow", alpha=0.25))

    for axT in (ax, ax2, ax3):
        ylim = axT.get_ylim()

        # ax.add_patch(plt.Rectangle((97, 300), 11, 600, facecolor="grey",alpha=0.25))

        # axT.set_xlim(min(P)/133.32,max(P)/133.32)
        axT.spines['top'].set_visible(False)
        axT.tick_params(axis='x', top='off')
        axT.spines['right'].set_visible(False)
        # axT.spines['bottom'].set_visible(False)
        axT.tick_params(axis='y', right='off')

    if fullRange == True:
        plt.savefig('WallmodelsDeterministicFullRange.pdf')
    else:
        plt.savefig('WallmodelsDeterministic.pdf')
        # plt.show()


if __name__ == "__main__":
    deterministicWallModelsQuantitaticePlot()
    exit()

    deterministicWallModels()
    exit()

    order = 4

    filename = ''.join(['laplaceHayashiReymond_wallmodel_order_C_A', str(order), '_.hdf5'])
    saveFile = h5py.File(filename, 'w')

    LaplaceGroup = saveFile.create_group('Laplace')
    dists, distsMeta = distributions('Hayashi')
    gpc(dists, distsMeta, LaplaceModel, order, hdf5group=LaplaceGroup, sampleScheme='H')

    HayashiGroup = saveFile.create_group('Hayashi')
    dists, distsMeta = distributions('Hayashi')
    gpc(dists, distsMeta, HayashiModel, order, hdf5group=HayashiGroup, sampleScheme='H')

    ReymondGroup = saveFile.create_group('Reymond')
    dists, distsMeta = distributions('Hayashi')
    gpc(dists, distsMeta, ReymondModel, order, hdf5group=ReymondGroup, sampleScheme='H')

    ReymondGroup10 = saveFile.create_group('Reymond-parameters-10%')
    dists, distsMeta = distributions('Reymond', devPar=0.10)
    gpc(dists, distsMeta, ReymondModel, order, hdf5group=ReymondGroup10, sampleScheme='H')

    # ReymondGroup = saveFile.create_group('Reymond-parameters-20%')
    # dists,distsMeta = distributions('Reymond', devPar = 0.2)
    # gpc(dists, distsMeta, ReymondModel, order, hdf5group= ReymondGroup, sampleScheme ='M')

    saveFile.flush()
    saveFile.close()

    print "finished"

# rerun = False
#     # filename = 'UQSA_data_gpc4_MC10.p'
#     # filename = 'UQSA_data.p'
#     # filename = 'UQSA_data5000mm.p'
#     # filename = 'UQ_MRI_data5000mm.p'
#     # filename = 'UQ_Invasive_P_data5000mm.p'
#     # filename = 'UQSA_dataJ.p'
#     filename = 'UQ_TestSubjects_data5000mm.p'
#     compare =  ['UQSA_data5000mm.p', 'UQ_TestSubjects_data5000mm.p']
#
#     dists,distsMeta = distributions()
#     if rerun:
#         print "estimating data"
#
#         solData = []
#
#         order = 4
#         sampleScheme ='S'
#         do_gpc = True
#         if do_gpc:
#             sol = gpc(dists,distsMeta, order, sampleScheme)
#             sol['case'] = "gpc4"
#             solData.append(sol)
#
#
#         numberOfSamples = 5000
#         sampleScheme ='L'
#         # solMC = mc(dists,distsMeta, numberOfSamples, sampleScheme)
#         solMC = mc_uq(dists,distsMeta, numberOfSamples, sampleScheme)
#         solMC['case'] = 'MC-'+str(numberOfSamples)
#         solData.append(solMC)
#
#         with open(filename,'wb') as pfile:
#             pickle.dump(solData,pfile)
#
#         # plot(solData,distsMeta)
#     elif compare:
#         solData =[]
#         for idx, filename in enumerate(compare):
#             with open(filename,'rb') as pfile:
#                 solData.append(pickle.load(pfile))
#         plotUQCmp(solData,distsMeta)
#
#     else:
#         with open(filename,'rb') as pfile:
#             solData = pickle.load(pfile)
#
#         plotUQ(solData,distsMeta)
#

