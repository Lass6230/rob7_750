
import numpy as np
from numpy import linalg
from scipy.optimize import linprog, minimize_scalar
from cvxopt import matrix, spmatrix
from cvxopt.solvers import options, qp, lp, conelp, coneqp
from scipy.stats import norm, chi2

# import matplotlib.pyplot as plt
# import matplotlib.lines as line
# from matplotlib import rc
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# rc('text', usetex=True)
import numdifftools as nd
from time import time

from typing import Callable
from dataclasses import dataclass

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import numpy as np
import matplotlib.animation as animation
import LB_optimizer as LB
import time

sim = LB.Simulation()