#!/usr/bin/env python
import re
from sys import argv, stdout, stderr, exit, modules
from optparse import OptionParser

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

from HiggsAnalysis.CombinedLimit.tfscipyhess import ScipyTROptimizerInterface

parser = OptionParser(usage="usage: %prog [options] datacard.txt -o output \nrun with --help to get list of options")
parser.add_option("-t","--toys", default=0, type=int, help="run a given number of toys, 0 fits the data (default), and -1 fits the asimov toy")
parser.add_option("","--toysFrequentist", default=True, action='store_true', help="run frequentist-type toys by randomizing constraint minima")
parser.add_option("","--bypassFrequentistFit", default=True, action='store_true', help="bypass fit to data when running frequentist toys to get toys based on prefit expectations")
parser.add_option("","--bootstrapData", default=False, action='store_true', help="throw toys directly from observed data counts rather than expectation from templates")
parser.add_option("","--randomizeStart", default=False, action='store_true', help="randomize starting values for fit (only implemented for asimov dataset for now")
parser.add_option("","--tolerance", default=1e-3, type=float, help="convergence tolerance for minimizer")
parser.add_option("","--expectSignal", default=1., type=float, help="rate multiplier for signal expectation (used for fit starting values and for toys)")
parser.add_option("","--seed", default=123456789, type=int, help="random seed for toys")
parser.add_option("","--fitverbose", default=0, type=int, help="verbosity level for fit")
parser.add_option("","--minos", default=[], type="string", action="append", help="run minos on the specified variables")
parser.add_option("","--scan", default=[], type="string", action="append", help="run likelihood scan on the specified variables")
parser.add_option("","--scanPoints", default=16, type=int, help="default number of points for likelihood scan")
parser.add_option("","--scanRange", default=3., type=float, help="default scan range in terms of hessian uncertainty")
parser.add_option("","--nThreads", default=-1., type=int, help="set number of threads (default is -1: use all available cores)")
(options, args) = parser.parse_args()

if len(args) == 0:
    parser.print_usage()
    exit(1)
    
seed = options.seed
print(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

options.fileName = args[0]

tf.train.import_meta_graph(options.fileName)

variables = tf.global_variables()

graph = tf.get_default_graph()
l = graph.get_tensor_by_name("loss:0")
rtheta = filter(lambda x: x.name == 'rtheta:0', variables)[0]
r = graph.get_tensor_by_name("r:0")
theta = graph.get_tensor_by_name("theta:0")
theta0 = filter(lambda x: x.name == 'theta0:0', variables)[0]
nexp = graph.get_tensor_by_name("nexp:0")
nexpnom = graph.get_tensor_by_name("nexpnom:0")
nobs = filter(lambda x: x.name == 'nobs:0', variables)[0]
pmaskedexp = graph.get_tensor_by_name("pmaskedexp:0")
maskedexp = graph.get_tensor_by_name("maskedexp:0")
pmaskedexpnorm = graph.get_tensor_by_name("pmaskedexpnorm:0")

cprocs = graph.get_tensor_by_name("cprocs:0")
csignals = graph.get_tensor_by_name("csignals:0")
csysts = graph.get_tensor_by_name("csysts:0")
cmaskedchans = graph.get_tensor_by_name("cmaskedchans:0")

dtype = rtheta.dtype.as_numpy_dtype
npoi = csignals.shape[0]
nsyst = csysts.shape[0]


grads = tf.gradients(l,rtheta)
grads = tf.identity(grads,"loss_grads")

grad = grads[0]

#uncertainty computation
hess = tf.hessians(l,rtheta)[0]
hess = tf.identity(hess,name="loss_hessian")

hessinv = tf.matrix_inverse(hess)


l0 = tf.Variable(np.zeros([],dtype=dtype),trainable=False)
x0 = tf.Variable(np.zeros(rtheta.shape,dtype=dtype),trainable=False)
a = tf.Variable(np.zeros([],dtype=dtype),trainable=False)
errdir = tf.Variable(np.zeros(rtheta.shape,dtype=dtype),trainable=False)

errproj = -tf.reduce_sum((rtheta-x0)*errdir,axis=0)

dxconstraint = a + errproj
dlconstraint = -(l - l0)
#gconstraint = grad - tf.reduce_sum(grad*errdir,axis=0)/tf.sqrt(tf.reduce_sum(tf.square(errdir),axis=0))
gconstraint = 1. - tf.reduce_sum(grad*errdir,axis=0)/tf.sqrt(tf.reduce_sum(tf.square(errdir),axis=0))/tf.sqrt(tf.reduce_sum(tf.square(grad),axis=0))

lb = np.concatenate((0.*np.ones([npoi],dtype=dtype),-np.inf*np.ones([nsyst],dtype=dtype)),axis=0)
ub = np.concatenate((np.inf*np.ones([npoi],dtype=dtype),np.inf*np.ones([nsyst],dtype=dtype)),axis=0)


globalinit = tf.global_variables_initializer()
nexpnomassign = tf.assign(nexpnom,nexp)
asimovassign = tf.assign(nobs,nexp)
asimovrandomizestart = tf.assign(rtheta,tf.clip_by_value(tf.contrib.distributions.MultivariateNormalFullCovariance(rtheta,hessinv).sample(),lb,ub))
bootstrapassign = tf.assign(nobs,tf.random_poisson(nobs,shape=[],dtype=dtype))
toyassign = tf.assign(nobs,tf.random_poisson(nexp,shape=[],dtype=dtype))
frequentistassign = tf.assign(theta0,theta + tf.random_normal(shape=theta.shape,dtype=dtype))
thetastartassign = tf.assign(rtheta, tf.concat([r,theta0],axis=0))
bayesassign = tf.assign(rtheta, tf.concat([r,theta+tf.random_normal(shape=theta.shape,dtype=dtype)],axis=0))


xtol = np.finfo(dtype).eps
btol = 1e-8
minimizer = ScipyTROptimizerInterface(l, var_list = [rtheta], var_to_bounds={rtheta: (lb,ub)}, options={'verbose': options.fitverbose, 'maxiter' : 100000, 'gtol' : 0., 'xtol' : xtol, 'barrier_tol' : btol})
minimizerscan = ScipyTROptimizerInterface(l, var_list = [rtheta], var_to_bounds={rtheta: (lb,ub)},  equalities=[dxconstraint], options={'verbose': options.fitverbose, 'maxiter' : 100000, 'gtol' : 0., 'xtol' : xtol, 'barrier_tol' : btol})
#minimizerminos = ScipyTROptimizerInterface(errproj,var_list = [rtheta], inequalities=[dlconstraint], options={'verbose': options.fitverbose, 'maxiter' : 100000, 'gtol' : 0., 'xtol' : xtol, 'barrier_tol' : 0.})
#minimizerminos = ScipyTROptimizerInterface(l,var_list = [rtheta], equalities=[gconstraint], inequalities=[dlconstraint], options={'verbose': options.fitverbose, 'maxiter' : 100000, 'gtol' : 0., 'xtol' : xtol, 'barrier_tol' : 0.})
#minimizerminos = ScipyTROptimizerInterface(errproj,var_list = [rtheta], var_to_bounds={rtheta: (lb,ub)}, equalities=[dlconstraint], options={'verbose': options.fitverbose, 'maxiter' : 100000, 'gtol' : 0., 'xtol' : xtol, 'barrier_tol' : 0.}, hess = lambda x : np.zeros(hess.shape,dtype=dtype))
minimizerminos = ScipyTROptimizerInterface(errproj,var_list = [rtheta], var_to_bounds={rtheta: (lb,ub)}, equalities=[dlconstraint], options={'verbose': options.fitverbose, 'maxiter' : 100000, 'gtol' : 0., 'xtol' : xtol, 'barrier_tol' : btol,})

#minimizerminos = ScipyTROptimizerInterface(tf.zeros([],dtype=dtype),var_list = [rtheta], var_to_bounds={rtheta: (lb,ub)}, equalities=[dlconstraint,gconstraint], options={'verbose': options.fitverbose, 'maxiter' : 100000, 'gtol' : 0., 'xtol' : xtol, 'barrier_tol' : btol})



#initialize tf session
if options.nThreads>0:
  config = tf.ConfigProto(intra_op_parallelism_threads=options.nThreads, inter_op_parallelism_threads=options.nThreads)
else:
  config = None

sess = tf.Session(config=config)
sess.run(globalinit)

rthetav = np.concatenate((options.expectSignal*np.ones([npoi],dtype=dtype), np.zeros([nsyst],dtype=dtype)), axis=0)
data_obs = sess.run(nobs)
procs, signals, systs, maskedchans = sess.run([cprocs,csignals,csysts,cmaskedchans])
signals = signals.tolist()
systs = systs.tolist()

pmaskedexpvals = sess.run(pmaskedexp)

#initialize output tree
f = ROOT.TFile( 'fitresults_%i.root' % seed, 'recreate' )
tree = ROOT.TTree("fitresults", "fitresults")

tseed = array('i', [seed])
tree.Branch('seed',tseed,'seed/I')

titoy = array('i',[0])
tree.Branch('itoy',titoy,'itoy/I')

tstatus = array('i',[0])
tree.Branch('status',tstatus,'status/I')

terrstatus = array('i',[0])
tree.Branch('errstatus',terrstatus,'errstatus/I')

tscanidx = array('i',[0])
tree.Branch('scanidx',tscanidx,'scanidx/I')

tedmval = array('f',[0.])
tree.Branch('edmval',tedmval,'edmval/F')

tnllval = array('f',[0.])
tree.Branch('nllval',tnllval,'nllval/F')

tdnllval = array('f',[0.])
tree.Branch('dnllval',tdnllval,'dnllval/F')

tchisq = array('f',[0.])
tree.Branch('chisq', tchisq, 'chisq/F')

tchisqpartial = array('f',[0.])
tree.Branch('chisqpartial', tchisqpartial, 'chisqpartial/F')

tndof = array('i',[0])
tree.Branch('ndof',tndof,'ndof/I')

tndofpartial = array('i',[0])
tree.Branch('ndofpartial',tndofpartial,'ndofpartial/I')

tsigvals = []
tsigerrs = []
tsigminosups = []
tsigminosdowns = []
tsiggenvals = []
for sig in signals:
  tsigval = array('f', [0.])
  tsigerr = array('f', [0.])
  tsigminosup = array('f', [0.])
  tsigminosdown = array('f', [0.])
  tsiggenval = array('f', [0.])
  tsigvals.append(tsigval)
  tsigerrs.append(tsigerr)
  tsigminosups.append(tsigminosup)
  tsigminosdowns.append(tsigminosdown)
  tsiggenvals.append(tsiggenval)
  tree.Branch(sig, tsigval, '%s/F' % sig)
  tree.Branch('%s_err' % sig, tsigerr, '%s_err/F' % sig)
  tree.Branch('%s_minosup' % sig, tsigminosup, '%s_minosup/F' % sig)
  tree.Branch('%s_minosdown' % sig, tsigminosdown, '%s_minosdown/F' % sig)
  tree.Branch('%s_gen' % sig, tsiggenval, '%s_gen/F' % sig)

tpmaskedexps = []
tpmaskedexpnorms = []
if len(maskedchans)>0:
  for proc,pmaskedexpval in zip(procs,pmaskedexpvals):
    tpmaskedexp = array('f', [1.])
    tpmaskedexpnorm = array('f', [1.])
    tpmaskedexps.append(tpmaskedexp)
    tpmaskedexpnorms.append(tpmaskedexpnorm)
    if pmaskedexpval > 0.:
      tree.Branch('%s_pmaskedexp' % proc, tpmaskedexp, '%s_pmaskedexp/F' % proc)
      tree.Branch('%s_pmaskedexpnorm' % proc, tpmaskedexpnorm, '%s_pmaskedexpnorm/F' % proc)

tthetavals = []
ttheta0vals = []
tthetaerrs = []
tthetaminosups = []
tthetaminosdowns = []
tthetagenvals = []
for syst in systs:
  systname = syst
  tthetaval = array('f', [0.])
  ttheta0val = array('f', [0.])
  tthetaerr = array('f', [0.])
  tthetaminosup = array('f', [0.])
  tthetaminosdown = array('f', [0.])
  tthetagenval = array('f', [0.])
  tthetavals.append(tthetaval)
  ttheta0vals.append(ttheta0val)
  tthetaerrs.append(tthetaerr)
  tthetaminosups.append(tthetaminosup)
  tthetaminosdowns.append(tthetaminosdown)
  tthetagenvals.append(tthetagenval)
  tree.Branch(systname, tthetaval, '%s/F' % systname)
  tree.Branch('%s_In' % systname, ttheta0val, '%s_In/F' % systname)
  tree.Branch('%s_err' % systname, tthetaerr, '%s_err/F' % systname)
  tree.Branch('%s_minosup' % systname, tthetaminosup, '%s_minosup/F' % systname)
  tree.Branch('%s_minosdown' % systname, tthetaminosdown, '%s_minosdown/F' % systname)
  tree.Branch('%s_gen' % systname, tthetagenval, '%s_gen/F' % systname)

tmaskedexps = []
for maskedchan in maskedchans:
  tmaskedexp = array('f',[0.])
  tmaskedexps.append(tmaskedexp)
  tree.Branch('%s_maskedexp' % maskedchan, tmaskedexp, '%s_maskedexp/F' % maskedchan)

ntoys = options.toys
if ntoys <= 0:
  ntoys = 1

#set likelihood offset
sess.run(nexpnomassign)

#prefit to data if needed
if options.toys>0 and options.toysFrequentist and not options.bypassFrequentistFit:  
  sess.run(nexpnomassign)
  ret = minimizer.minimize(sess)
  rthetav = sess.run(rtheta)

def printfunc(l):
  print(l)

def printstep(x,y):
  print([x,y])

for itoy in range(ntoys):
  titoy[0] = itoy

  #reset all variables
  sess.run(globalinit)
  rtheta.load(rthetav,sess)
  xvalgen = rthetav
    
  dofit = True
  
  if options.toys < 0:
    print("Running fit to asimov toy")
    sess.run(asimovassign)
    if options.randomizeStart:
      sess.run(asimovrandomizestart)
    else:
      dofit = False
  elif options.toys == 0:
    print("Running fit to observed data")
    nobs.load(data_obs,sess)
  else:
    print("Running toy %i" % itoy)  
    if options.toysFrequentist:
      #randomize nuisance constraint minima
      sess.run(frequentistassign)
    else:
      #randomize actual values
      sess.run(bayesassign)      
      
    if options.bootstrapData:
      #randomize from observed data
      nobs.load(data_obs,sess)
      sess.run(bootstrapassign)
      xvalgen = -1.*np.ones_like(xvalgen)
    else:
      xvalgen = sess.run(rtheta)
      #randomize from expectation
      sess.run(toyassign)      

  #assign start values for nuisance parameters to constraint minima
  sess.run(thetastartassign)
  #set likelihood offset
  sess.run(nexpnomassign)
  if dofit:
    ret = minimizer.minimize(sess)

  xval, nllval, gradval, hessval = sess.run([rtheta,l,grad,hess])
  dnllval = 0.

  isposdef = np.all(np.greater_equal(np.linalg.eigvalsh(hessval),0.))
    
  #get fit values
  rvals = xval[:npoi]
  thetavals = xval[npoi:]  
  
  theta0vals = sess.run(theta0)
  
  pmaskedexpvals = sess.run(pmaskedexp)
  pmaskedexpnormvals = sess.run(pmaskedexpnorm)
  maskedexpvals = sess.run(maskedexp)
  
  hessvaltrans = hessval
  
  rvalsgen = np.exp(xvalgen[:npoi])
  thetavalsgen = xvalgen[npoi:]
  xvalgentrans = np.concatenate((rvalsgen,thetavalsgen),axis=0)
    
  dx = xvalgen - xval
  dx = np.reshape(dx,[-1,1])
  
  chisq = np.matmul(np.transpose(dx),np.matmul(hessval,dx))    
  chisqpartial = np.matmul(np.transpose(dx[:npoi]),np.matmul(hessval[:npoi,:npoi],dx[:npoi]))
      
  try:        
    invhess = np.linalg.inv(hessval)
    edmval = 0.5*np.matmul(np.matmul(np.transpose(gradval),invhess),gradval)
    
    rawsigmasv = np.sqrt(np.diag(invhess))
    sigmasv = rawsigmasv
    
    errstatus = 0
    if np.any(np.isnan(sigmasv)):
      errstatus = 1
  except np.linalg.LinAlgError:
    edmval = -99.
    sigmasv = -99.*np.ones_like(xval)
    errstatus = 2
  
  if isposdef and edmval > 0.:
    status = 0
  else:
    status=1
    
  print("status = %i, errstatus = %i, nllval = %f, edmval = %e" % (status,errstatus,nllval,edmval))
  
  
  minoserrsup = -99.*np.ones_like(sigmasv)
  minoserrsdown = -99.*np.ones_like(sigmasv)
  for var in options.minos:
    print("running minos-like algorithm for %s" % var)
    if var in signals:
      erridx = signals.index(var)
    elif var in systs:
      erridx = npoi + systs.index(var)
    else:
      raise Exception()
      
    l0.load(nllval+0.5,sess)
    x0.load(xval,sess)

    errdirv = np.zeros_like(rthetav)
    
    errdirv[erridx] = 1./rawsigmasv[erridx]
    errdir.load(errdirv,sess)
    rtheta.load(xval + 1.*rawsigmasv[erridx]*rawsigmasv[erridx]*errdirv,sess)
    a.load(1.,sess)
    minimizerscan.minimize(sess)
    #rtheta.load(xval,sess)
    #minimizerminosalt.minimize(sess)
    minimizerminos.minimize(sess)
    xvalminosup, nllvalminosup = sess.run([rtheta,l])
    dxvalup = xvalminosup[erridx]-xval[erridx]
    minoserrsup[erridx] = dxvalup

    errdirv[erridx] = -1./rawsigmasv[erridx]
    errdir.load(errdirv,sess)
    rtheta.load(xval + 1.*rawsigmasv[erridx]*rawsigmasv[erridx]*errdirv,sess)
    #rtheta.load(xval,sess)
    a.load(1.,sess)
    minimizerscan.minimize(sess)
    #minimizerminosalt.minimize(sess)
    minimizerminos.minimize(sess)
    xvalminosdown, nllvalminosdown = sess.run([rtheta,l])
    dxvaldown = -(xvalminosdown[erridx]-xval[erridx])
    minoserrsdown[erridx] = dxvaldown
    
  tstatus[0] = status
  terrstatus[0] = errstatus
  tedmval[0] = edmval
  tnllval[0] = nllval
  tdnllval[0] = dnllval
  tscanidx[0] = -1
  tchisq[0] = chisq
  tchisqpartial[0] = chisqpartial
  tndof[0] = xval.shape[0]
  tndofpartial[0] = npoi
  
  rsigmasv = sigmasv[:npoi]
  thetasigmasv = sigmasv[npoi:]
  
  rminosups = minoserrsup[:npoi]
  rminosdowns = minoserrsdown[:npoi]
  
  thetaminosups = minoserrsup[npoi:]
  thetaminosdowns = minoserrsdown[npoi:]
    
  for sig,sigval,sigma,minosup,minosdown,siggenval,tsigval,tsigerr,tsigminosup,tsigminosdown,tsiggenval in zip(signals,rvals,rsigmasv,rminosups,rminosdowns,rvalsgen,tsigvals,tsigerrs,tsigminosups,tsigminosdowns,tsiggenvals):
    tsigval[0] = sigval
    tsigerr[0] = sigma
    tsigminosup[0] = minosup
    tsigminosdown[0] = minosdown
    tsiggenval[0] = siggenval
    if itoy==0:
      print('%s = %e +- %f (+%f -%f)' % (sig,sigval,sigma,minosup,minosdown))

  for proc,pmaskedexpval,pmaskedexpnormval, tpmaskedexp, tpmaskedexpnorm in zip(procs,pmaskedexpvals,pmaskedexpnormvals,tpmaskedexps,tpmaskedexpnorms):
    tpmaskedexp[0] = pmaskedexpval
    tpmaskedexpnorm[0] = pmaskedexpnormval
    
  for maskedchan,maskedexpval,tmaskedexp in zip(maskedchans,maskedexpvals,tmaskedexps):
    tmaskedexp[0] = maskedexpval

  for syst,thetaval,theta0val,sigma,minosup,minosdown,thetagenval, tthetaval,ttheta0val,tthetaerr,tthetaminosup,tthetaminosdown,tthetagenval in zip(systs,thetavals,theta0vals,thetasigmasv,thetaminosups,thetaminosdowns,thetavalsgen, tthetavals,ttheta0vals,tthetaerrs,tthetaminosups,tthetaminosdowns,tthetagenvals):
    tthetaval[0] = thetaval
    ttheta0val[0] = theta0val
    tthetaerr[0] = sigma
    tthetaminosup[0] = minosup
    tthetaminosdown[0] = minosdown
    tthetagenval[0] = thetagenval
    if itoy==0:
      print('%s = %f +- %f (+%f -%f) (%s_In = %f)' % (syst, thetaval, sigma, minosup,minosdown,syst,theta0val))
    
  tree.Fill()
  
  for var in options.scan:
    print("running profile likelihood scan for %s" % var)
    if var in signals:
      isSig = True
      erridx = signals.index(var)
    elif var in systs:
      isSig = False
      erridx = npoi + systs.index(var)
    else:
      raise Exception()
    
    x0.load(xval,sess)
    
    errdirv = np.zeros_like(rthetav)
    errdirv[erridx] = 1./rawsigmasv[erridx]
    errdir.load(errdirv,sess)
        
    dsigs = np.linspace(0.,options.scanRange,options.scanPoints)
    signs = [1.,-1.]
    
    for sign in signs:
      rtheta.load(xval,sess)
      for absdsig in dsigs:
        dsig = sign*absdsig
        
        if absdsig==0. and sign==-1.:
          continue
        
        a.load(dsig,sess)
        minimizerscan.minimize(sess)
    
        xvalscan, nllvalscan = sess.run([rtheta,l])
        dnllvalscan = nllvalscan - nllval
          
        #get fit values
        rvals = xvalscan[:npoi]
        thetavals = xvalscan[npoi:]  
        
        pmaskedexpvals = sess.run(pmaskedexp)
        pmaskedexpnormvals = sess.run(pmaskedexpnorm)
        maskedexpvals = sess.run(maskedexp)
        
        tscanidx[0] = erridx
        tnllval[0] = nllvalscan
        tdnllval[0] = dnllvalscan
        
        for sig,sigval,sigma,minosup,minosdown,tsigval,tsigerr,tsigminosup,tsigminosdown in zip(signals,rvals,rsigmasv,rminosups,rminosdowns,tsigvals,tsigerrs,tsigminosups,tsigminosdowns):
          tsigval[0] = sigval
          
        for proc,pmaskedexpval,pmaskedexpnormval, tpmaskedexp, tpmaskedexpnorm in zip(procs,pmaskedexpvals,pmaskedexpnormvals,tpmaskedexps,tpmaskedexpnorms):
          tpmaskedexp[0] = pmaskedexpval
          tpmaskedexpnorm[0] = pmaskedexpnormval
          
        for maskedchan,maskedexpval,tmaskedexp in zip(maskedchans,maskedexpvals,tmaskedexps):
          tmaskedexp[0] = maskedexpval
        
        for syst,thetaval,theta0val,sigma,minosup,minosdown,tthetaval,ttheta0val,tthetaerr,tthetaminosup,tthetaminosdown in zip(systs,thetavals,theta0vals,thetasigmasv,thetaminosups,thetaminosdowns, tthetavals,ttheta0vals,tthetaerrs,tthetaminosups,tthetaminosdowns):
          tthetaval[0] = thetaval

        tree.Fill()


f.Write()
f.Close()
