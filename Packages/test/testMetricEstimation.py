# test command generation and config for metricEstimation
# run this in following ways. unittest is more correct
# TODO setup tests correctly ala unittest to see output
# python3 -m unittest test/testMetricEstimation.py
#
# to avoid unittest, run from top level metpy directory as
# PYTHONPATH=. python3 test/testMetricEstimation.py
from apps import metricEstimation
from util import YAMLcfg

def testConfig():
  print("Testing metricEstimation config")
  cfg = metricEstimation.default_config()
  cfstr = YAMLcfg.ConfigToYAML(metricEstimation.MetricEstimatorConfigSpec, cfg)
  print("Default Config:\n",cfstr)
  return(cfg)

def testEucInit(config):
  print("\n\nTesting metricEstimation with Euclidean initialization. Note that metric estimation won't actually run, we just want to confirm that the generated commands are correct.")
  config.options.do_Euclidean_initialization = True
  me = metricEstimation.MetricEstimator(config)
  me.construct_commands()
  print(me.commands)

def testNoEucInit(config):
  print("\n\nTesting metricEstimation without Euclidean initialization. Note that metric estimation won't actually run, we just want to confirm that the generated commands are correct.")
  config.options.do_Euclidean_initialization = False
  me = metricEstimation.MetricEstimator(config)
  me.construct_commands()
  print(me.commands)


if __name__ == "__main__":
  cfg = testConfig()
  cfg.paths.commandPrefix = '~/Software/AdaptaMetric/build'
  cfg.paths.inputPrefix = '/usr/sci/projects/HCP/Kris/NSFCRCNS/TestData'
  cfg.paths.outputPrefix = '/usr/sci/projects/HCP/Kris/NSFCRCNS/TestResults/testMetricEstimation'
  cfg.options.dim = 2
  cfstr = YAMLcfg.ConfigToYAML(metricEstimation.MetricEstimatorConfigSpec, cfg)
  print("Modified Config:\n",cfstr)
  testEucInit(cfg)
  testNoEucInit(cfg)
