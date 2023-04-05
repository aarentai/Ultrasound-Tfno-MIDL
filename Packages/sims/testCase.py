# class to support running a test case in a simulation

class TestCase():
  def __init__(self, name, config):
    self.name = name
    self.config = config
    self.results = None
    self.has_run = False

class TestCases():
  def __init__(self):
    self.cases = {}

  def add_test_case(self, name, config):
    self.cases[name] = TestCase(name, config)
