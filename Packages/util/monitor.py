# ======================================================================
# Copyright (c) 2020, Scientific Computing and Imaging Institute,
# University of Utah. All rights reserved.
# Author: Kris Campbell
# License: New BSD 3-Clause (see accompanying LICENSE file for details)
# ======================================================================
# tools for monitoring progress
# includes logging, timers, checkpoints etc.

# python imports
import time

# Timing utilities
class Timer():
    """ Timer class is a utility to measure running time.  

    start will start the timer
    pause will pause the timer and accumulate the time to run so far
    resume will start the timer again 
    reset will clear the timer and start over
    timesofar will prvode the accumulated run time so far across all pauses since the last reset
    """
    def __init__(self, auto_start=True):
        self.start_time = 0
        self.stop = 1
        self.cum_time = 0
        if auto_start:
            self.start()
    
    def start(self):
        self.resume()

    def pause(self):
        self.stop = time.time() - self.start_time
        self.cum_time += self.stop

    def resume(self):
        self.start_time = time.time()
        self.stop = 0

    def reset(self):
        self.start_time = time.time()
        self.cum_time = 0
        self.stop = 0

    def timesofar(self):
        # if we're not paused, return accumulated time + current time
        # if we are paused, only return accumulated time
        if not self.stop:
            return time.time() - self.start_time + self.cum_time
        else:
            return self.cum_time
# end class Timer
