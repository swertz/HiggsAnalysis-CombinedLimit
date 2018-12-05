#!/usr/bin/env python
import re
from sys import argv, stdout, stderr, exit, modules
from optparse import OptionParser
from copy import deepcopy

import tensorflow as tf
import numpy as np
import scipy

import math



# import ROOT with a fix to get batch mode (http://root.cern.ch/phpBB3/viewtopic.php?t=3198)
argv.append( '-b-' )
import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.IgnoreCommandLineOptions = True
argv.remove( '-b-' )

from array import array

from HiggsAnalysis.CombinedLimit.DatacardParser import *
from HiggsAnalysis.CombinedLimit.ModelTools import *
from HiggsAnalysis.CombinedLimit.ShapeTools import *
from HiggsAnalysis.CombinedLimit.PhysicsModel import *
from HiggsAnalysis.CombinedLimit.tfscipyhess import ScipyTROptimizerInterface

parser = OptionParser(usage="usage: %prog [options] datacard.txt -o output \nrun with --help to get list of options")
addDatacardParserOptions(parser)
parser.add_option("-P", "--physics-model", dest="physModel", default="HiggsAnalysis.CombinedLimit.PhysicsModel:defaultModel",    type="string", help="Physics model to use. It should be in the form (module name):(object name)")
parser.add_option("--PO", "--physics-option", dest="physOpt", default=[],    type="string", action="append", help="Pass a given option to the physics model (can specify multiple times)")
parser.add_option("", "--dump-datacard", dest="dumpCard", default=False, action='store_true',    help="Print to screen the DataCard as a python config and exit")
parser.add_option("","--allowNegativeExpectation", default=False, action='store_true', help="allow negative expectation")
parser.add_option("","--POIMode", default="mu",type="string", help="mode for POI's")
parser.add_option("","--nonNegativePOI", default=False, action='store_true', help="force signal strengths to be non-negative")
parser.add_option("","--POIDefault", default=1., type=float, help="mode for POI's")
parser.add_option("","--maskedChan", default=[], type="string",action="append", help="channels to be masked in likelihood but propagated through for later storage/analysis")
parser.add_option("-S","--doSystematics", type=int, default=1, help="enable systematics")
(options, args) = parser.parse_args()

if len(args) == 0:
    parser.print_usage()
    exit(1)

if options.nonNegativePOI:
    boundmode = 1
else:
    boundmode = 0

options.fileName = args[0]
if options.fileName.endswith(".gz"):
    import gzip
    file = gzip.open(options.fileName, "rb")
    options.fileName = options.fileName[:-3]
else:
    file = open(options.fileName, "r")

## Parse text file
DC = parseCard(file, options)

if options.dumpCard:
    DC.print_structure()
    exit()

print(options)

nproc = len(DC.processes)
nsignals = len(DC.signals)

dtype = 'float64'

MB = ShapeBuilder(DC, options)

#list of processes, signals first
procs = []
for proc in DC.processes:
    if DC.isSignal[proc]:
        procs.append(proc)

for proc in DC.processes:
    if not DC.isSignal[proc]:
        procs.append(proc)

#list of signals preserving datacard order
signals = []
for proc in DC.processes:
    if DC.isSignal[proc]:
        signals.append(proc)

#list of systematic uncertainties (nuisances)
systs = []
if options.doSystematics:
    for syst in DC.systs:
        systs.append(syst[0])

nsyst = len(systs)

#list of channels, ordered such that masked channels are last
chans = []
for chan in DC.bins:
    if not chan in options.maskedChan:
        chans.append(chan)

maskedchans = []
for chan in DC.bins:
    if chan in options.maskedChan:
        chans.append(chan)
        maskedchans.append(chan)

#fill data, expected yields, and kappas

#n.b data and expected have shape [nbins]

#norm has shape [nbins,nprocs] and keeps track of expected normalization

#logkup/down have shape [nbins, nprocs, nsyst] and keep track of systematic variations
#per-bin, per-process, per nuisance-parameter

logkepsilon = math.log(1e-3)

nbinstotal = 0
nbinsmasked = 0
data_obs = np.empty([0],dtype=dtype)
norm = np.empty([0,nproc],dtype=dtype)
logkup = np.empty([0,nproc,nsyst],dtype=dtype)
logkdown = np.empty([0,nproc,nsyst],dtype=dtype)
for chan in chans:
    expchan = DC.exp[chan]

    #FIXME:    hack to run without observed data for masked channels
    if chan in options.maskedChan:
        nbins = 1
        nbinsmasked += nbins
    else:
        datahist = MB.getShape(chan,"data_obs")
        datanp = np.array(datahist).astype(dtype)[1:-1]
        data_obs = np.concatenate((data_obs,datanp))
        nbins = datanp.shape[0]

    normchan = np.empty([nbins,0],dtype=dtype)
    logkupchan = np.empty([nbins,0,nsyst],dtype=dtype)
    logkdownchan = np.empty([nbins,0,nsyst],dtype=dtype)
    for proc in procs:
        hasproc = proc in expchan

        if hasproc:
            normhist = MB.getShape(chan,proc)
            normnp = np.array(normhist).astype(dtype)[1:-1]
            normnp = np.reshape(normnp,[-1,1])
            if not options.allowNegativeExpectation:
                normnp = np.maximum(normnp,0.)

            if normnp.shape[0] != nbins:
                raise Exception("Error: number of bins in histogram does not match between expectation and observation")
        else:
            normnp = np.zeros([nbins,1],dtype=dtype)

        normchan = np.concatenate((normchan,normnp),axis=1)

        logkupproc = np.empty([nbins,1,0],dtype=dtype)
        logkdownproc = np.empty([nbins,1,0],dtype=dtype)
        for syst in DC.systs[:nsyst]:
            name = syst[0]
            stype = syst[2]

            if not hasproc:
                logkupsyst = np.zeros([nbins,1,1],dtype=dtype)
                logkdownsyst = np.zeros([nbins,1,1],dtype=dtype)
            elif stype=='lnN':
                ksyst = syst[4][chan][proc]
                if type(ksyst) is list:
                    ksystup = ksyst[1]
                    ksystdown = ksyst[0]
                    if ksystup == 0.:
                        ksystup = 1.
                    if ksystdown == 0.:
                        ksystdown = 1.
                    logkupsyst = math.log(ksystup)*np.ones([nbins,1,1],dtype=dtype)
                    logkdownsyst = -math.log(ksystdown)*np.ones([nbins,1,1],dtype=dtype)
                else:
                    if ksyst == 0.:
                        ksyst = 1.
                    logkupsyst = math.log(ksyst)*np.ones([nbins,1,1],dtype=dtype)
                    logkdownsyst = math.log(ksyst)*np.ones([nbins,1,1],dtype=dtype)

            elif 'shape' in stype:
                kfac = syst[4][chan][proc]

                if kfac>0:
                    normhistup = MB.getShape(chan,proc,name+"Up")
                    normnpup = np.array(normhistup).astype(dtype)[1:-1]
                    normnpup = np.reshape(normnpup,[-1,1])
                    if normnpup.shape[0] != nbins:
                        raise Exception("Error: number of bins in histogram does not match between nominal and systematic variation")
                    logkupsyst = kfac*np.log(normnpup/normnp)
                    logkupsyst = np.where(np.equal(np.sign(normnp*normnpup),1), logkupsyst, logkepsilon*np.ones_like(logkupsyst))
                    logkupsyst = np.reshape(logkupsyst,[-1,1,1])

                    normhistdown = MB.getShape(chan,proc,name+"Down")
                    normnpdown = np.array(normhistdown).astype(dtype)[1:-1]
                    normnpdown = np.reshape(normnpdown,[-1,1])
                    if normnpdown.shape[0] != nbins:
                        raise Exception("Error: number of bins in histogram does not match between nominal and systematic variation")
                    logkdownsyst = -kfac*np.log(normnpdown/normnp)
                    logkdownsyst = np.where(np.equal(np.sign(normnp*normnpdown),1), logkdownsyst, -logkepsilon*np.ones_like(logkdownsyst))
                    logkdownsyst = np.reshape(logkdownsyst,[-1,1,1])
                else:
                    logkupsyst = np.zeros([normnp.shape[0],1,1],dtype=dtype)
                    logkdownsyst = np.zeros([normnp.shape[0],1,1],dtype=dtype)
            else:
                raise Exception('Unsupported systematic type')

            logkupproc = np.concatenate((logkupproc,logkupsyst),axis=2)
            logkdownproc = np.concatenate((logkdownproc,logkdownsyst),axis=2)

        logkupchan = np.concatenate((logkupchan,logkupproc),axis=1)
        logkdownchan = np.concatenate((logkdownchan,logkdownproc),axis=1)

    norm = np.concatenate((norm,normchan), axis=0)

    logkup = np.concatenate((logkup,logkupchan),axis=0)
    logkdown = np.concatenate((logkdown,logkdownchan),axis=0)

    nbinstotal += nbins

#print(np.max(np.abs(logkup)))
#print(np.max(np.abs(logkdown)))

rateParams = {}
for chan in chans:
    rateParams[chan] = {}
    for proc in procs:
        rateParams[chan][proc] = {}
dependentRateParams = deepcopy(rateParams)

regex = re.compile(r'@([0-9]*)')
# def get_tf_variable(name):
    # return [var for var in tf.global_variables() if str(var.op.name) == name][0]

for rp in DC.rateParams.items():
    paramKey = rp[0]
    channel = paramKey.split('AND')[0]
    proc = paramKey.split('AND')[1]
    paramCfg = rp[1][0]
    
    # FIXME bounds not used at the momemt
    if paramCfg[1] == '':
        bounds = None
    else:
        bounds = list(paramCfg[1])
 
    paramCfg = paramCfg[0]
    name = paramCfg[0]

    if paramCfg[-1] == 0:
        # actual parameter
        default = float(paramCfg[1])
        rateParams[channel][proc][name] = default
        # print("Param {}: default value = {}".format(name, default))
        # rateParams[channel][proc][name] = tf.Variable(default, dtype=dtype, name=name)

    if paramCfg[-1] == 1:
        # dependent parameter
        formula = paramCfg[1]
        # formula = regex.sub(lambda x: 'get_tf_variable("{' + x.group(0).replace('@', '') + '}")', formula)
        formula = regex.sub(lambda x: 'graph.get_tensor_by_name("{' + x.group(0).replace('@', '') + '}:0")', formula)
        formulaInputs = paramCfg[2].split(',')

        dependentRateParams[channel][proc][name] = (formula, formulaInputs)

extArgs = {}
for rp in DC.extArgs.items():
    name = rp[0]
    # FIXME bounds not used at the momemt
    default = float(rp[1][2])
    extArgs[name] = default
    # print("Param {}: default value = {}".format(name, default))
    # extArgs[name] = tf.Variable(float(rp[1][2]), dtype=dtype, name=name)

# freeParams only contains standalone parameters to be fitted
freeParamDefault = extArgs.values()
freeParamNames = extArgs.keys()
for chan in chans:
    for proc in procs:
        for name, param in rateParams[chan][proc].items():
            freeParamDefault.append(param)
            freeParamNames.append(name)
nFree = len(freeParamNames)


## Create variables to fit

pois = []

if options.POIMode == "mu":
    # npoi = nsignals
    npoi = 1 # even if many signals, only one POI!
    poidefault = options.POIDefault * np.ones([npoi],dtype=dtype)
    # for signal in signals:
        # pois.append(signal)
    pois = ['mu']
elif options.POIMode == "none":
    npoi = 0
    poidefault = np.empty([],dtype=dtype)
else:
    raise Exception("unsupported POIMode")

nparams = npoi + nsyst + nFree
params = pois + systs + freeParamNames

if boundmode:
    xpoidefault = np.sqrt(poidefault)
else:
    xpoidefault = poidefault

print("nbins = %d, npoi = %d, nsyst = %d, nFree = %d" % (data_obs.shape[0], npoi, nsyst, nFree))

cprocs = tf.constant(procs,name="cprocs")
csignals = tf.constant(signals, name="csignals")
# csysts = tf.concat([tf.constant(systs), tf.constant(freeParamNames)], axis=0, name="csysts")
csysts = tf.constant(systs, name="csysts")
cfrees = tf.constant(freeParamNames, name="cfrees")
cmaskedchans = tf.constant(maskedchans, name="cmaskedchans")
cpois = tf.constant(pois, name="cpois")

# data
nobs = tf.Variable(data_obs, trainable=False, name="nobs")
# randomised nuisances
theta0 = tf.Variable(tf.zeros([nsyst], dtype=dtype), trainable=False, name="theta0")

# tf variable containing all fit parameters
thetadefault = tf.zeros([nsyst], dtype=dtype)
freeParamDefault = tf.constant(freeParamDefault, dtype=dtype, name="free0")
if npoi > 0:
    xdefault = tf.concat([xpoidefault, thetadefault, freeParamDefault], axis=0)
else:
    xdefault = tf.concact([thetadefault, freeParamDefault], axis=0)

x = tf.Variable(xdefault, name='x')

xpoi = x[:npoi]
theta = x[npoi:npoi+nsyst]
free = x[npoi+nsyst:]

if boundmode:
    poi = tf.square(xpoi)
    # not used in combine! FIXME
    jacpoitheta = tf.diag(tf.concat([2.*xpoi, tf.ones([nsyst+nFree],dtype=dtype)], axis=0))
else:
    poi = xpoi

xpoi = tf.identity(poi, name="xpoi")
poi = tf.identity(poi, name=options.POIMode)
theta = tf.identity(theta, name="theta")
free = tf.identity(free, name="free")

# Create aliases for the now-defined free parameter variables
freeParams = [ tf.identity(free[i], name=freeParamNames[i]) for i in range(nFree) ]

# Put those aliases back into the rateParams dictionary for later use, skipping the extArgs
counter = len(extArgs)
for chan in chans:
    for proc in procs:
        for name in rateParams[chan][proc].keys():
            # print("assigning {} to channel {}, proc {}, name {}".format(freeParams[counter], chan, proc, name))
            rateParams[chan][proc][name] = freeParams[counter]
            counter += 1

# Create dependent rate parameters based on free parameters and formulas
# and add them to the collection of rate parameters
graph = tf.get_default_graph() # needed when executing the formulas translated above
for chan in dependentRateParams.keys():
    for proc in dependentRateParams[chan].keys():
        for name, param in dependentRateParams[chan][proc].items():
            formulaInputs = param[1]
            formula = param[0].format(*formulaInputs)
            # print("attempting to execute: {}".format(formula))
            exec(name + "=" + formula)
            exec("{0} = tf.identity({0}, name='{0}')".format(name))
            # result = locals()[name]
            result = graph.get_tensor_by_name(name + ":0")
            # print("assigning {} to channel {}, proc {}, name {}".format(result, chan, proc, name))
            rateParams[chan][proc][name] = result

# Multiply rate parameters with the process and channel they correspond to (in all bins of that channel)
rateParamInfos = []
for chan in chans:
    for proc in procs:
        if len(rateParams[chan][proc]) > 1:
            raise Exception("Only one rate param per process per channel is supported at the moment")
        for p in rateParams[chan][proc].values():
            rateParamInfos.append((chan, proc, p))
rateParamTensor = tf.concat([[p[2]] for p in rateParamInfos], axis=0)
nRate = len(rateParamInfos)

rateParamMatrix = np.zeros((nbinstotal, nproc, nRate+1))
rateParamVector = tf.concat([tf.constant([1.], dtype=dtype), rateParamTensor], axis=0)
# rateParamVector = tf.reshape(rateParamVector, [-1, 1])

for iparam, param in enumerate(rateParamInfos):
    chan = param[0]
    proc = param[1]
    ichan = chans.index(chan)
    iproc = procs.index(proc)

    for ibin in range(nbins):
        globalBin = ichan * nbins + ibin
        rateParamMatrix[globalBin, iproc, iparam+1] = 1.
        # print("channel {}, bin {}, global bin {}, process {}, assign: {}".format(chan, ibin, globalBin, proc, param[2]))

for ichan in range(len(chans)):
    for ibin in range(nbins):
        globalBin = ichan * nbins + ibin
        for iproc in range(nproc):
            s = np.sum(rateParamMatrix[globalBin, iproc, :])
            if s > 1:
                raise Exception("something has gone wrong")
            if s == 0:
                rateParamMatrix[globalBin, iproc, 0] = 1.

# np.set_printoptions(threshold=np.nan)
# print(rateParamMatrix)

rateParamMatrix = tf.constant(rateParamMatrix, dtype=dtype)
# to be multiplied with expected numbers
rRate = tf.tensordot(rateParamMatrix, rateParamVector, axes=[[2], [0]])
# rRate = tf.reshape(rRate, [nbinstotal, nproc])

# print(rateParamInfos)
# print(freeParams)
# print(rateParams)
# print(dependentRateParams)
# print(extArgs)

# Interpolation for asymmetric log-normal

logkavg = 0.5*(logkup+logkdown)
logkhalfdiff = 0.5*(logkup-logkdown)

twox = 2.*theta
twox2 = twox*twox
alpha = 0.125 * twox * (twox2 * (3*twox2 - 10.) + 15.)
alpha = tf.clip_by_value(alpha,-1.,1.)
logk = logkavg + alpha*logkhalfdiff

#matrix encoding effect of nuisance parameters
logsnorm = tf.reduce_sum(logk * theta, axis=-1)
snorm = tf.exp(logsnorm)

#vector encoding effect of signal strengths
if options.POIMode == "mu":
    r = tf.tile(poi, [nsignals])
elif options.POIMode == "none":
    r = tf.ones([nsignals],dtype=dtype)

rnorm = tf.concat([r,tf.ones([nproc-nsignals],dtype=dtype)],axis=0)
rnorm = tf.reshape(rnorm,[1,-1])

#final expected yields per-bin including effect of signal
#strengths and nuisance parmeters
pnormfull = rnorm*snorm*norm*rRate
if nbinsmasked>0:
    pnorm = pnormfull[:nbinstotal-nbinsmasked]
else:
    pnorm = pnormfull

nexp = tf.reduce_sum(pnorm,axis=-1)
nexp = tf.identity(nexp,name='nexp')

nexpsafe = tf.where(tf.equal(nobs,tf.zeros_like(nobs)), tf.ones_like(nobs), nexp)
lognexp = tf.log(nexpsafe)

nexpnom = tf.Variable(nexp, trainable=False, name="nexpnom")
nexpnomsafe = tf.where(tf.equal(nobs,tf.zeros_like(nobs)), tf.ones_like(nobs), nexpnom)
lognexpnom = tf.log(nexpnomsafe)

#final likelihood computation

#poisson term
lnfull = tf.reduce_sum(-nobs*lognexp + nexp, axis=-1)

#poisson term with offset to improve numerical precision
ln = tf.reduce_sum(-nobs*(lognexp-lognexpnom) + nexp-nexpnom, axis=-1)

#constraints
lc = tf.reduce_sum(0.5*tf.square(theta - theta0))

l = ln + lc
l = tf.identity(l,name="loss")

# not used in combine! FIXME
lfull = lnfull + lc
lfull = tf.identity(lfull,name="lossfull")

pnormmasked = pnormfull[nbinstotal-nbinsmasked:,:nsignals]
pmaskedexp = tf.reduce_sum(pnormmasked, axis=0)
pmaskedexp = tf.identity(pmaskedexp, name="pmaskedexp")

maskedexp = tf.reduce_sum(pnormmasked, axis=-1,keepdims=True)
maskedexp = tf.identity(maskedexp,"maskedexp")

if nbinsmasked>0:
    pmaskedexpnorm = tf.reduce_sum(pnormmasked/maskedexp, axis=0)
else:
    pmaskedexpnorm = pmaskedexp
pmaskedexpnorm = tf.identity(pmaskedexpnorm,"pmaskedexpnorm")

outputs = []

outputs.append(poi)
if nbinsmasked>0:
    outputs.append(pmaskedexp)
    outputs.append(pmaskedexpnorm)

for output in outputs:
    tf.add_to_collection("outputs",output)

basename = '.'.join(options.fileName.split('.')[:-1])
if options.out:
        basename = options.out.replace('.meta','')
tf.train.export_meta_graph(filename='%s.meta' % basename)
