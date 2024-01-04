"""
Copyright, the CVXPY authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import scipy.sparse as sp

import cvxpy.interface as intf
import cvxpy.settings as s
from cvxpy.constraints import NonNeg, Zero
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver


class CUPDLP(ConicSolver):
    """An interface for the culp solver."""

    # Solver capabilities.
    MIP_CAPABLE = False
    SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS

    STATUS_MAP = {
        0: s.OPTIMAL,
        1: s.INFEASIBLE,
        2: s.UNBOUNDED,
        3: s.INFEASIBLE_OR_UNBOUNDED,
        4: s.USER_LIMIT,    # TIMELIMIT_OR_ITERLIMIT
    }

    def import_solver(self) -> None:
        """Imports the solver."""
        import culpy  # noqa F401

    def name(self):
        """The name of the solver."""
        return s.CUPDLP

    def apply(self, problem):
        """Returns a new problem and data for inverting the new solution.

        Returns
        -------
        tuple
            (dict of arguments needed for the solver, inverse data)
        """
        data = {}
        inv_data = {self.VAR_ID: problem.x.id}

        # Format constraints
        if not problem.formatted:
            problem = self.format_constraints(problem, self.EXP_CONE_ORDER)
        data[s.PARAM_PROB] = problem
        data[self.DIMS] = problem.cone_dims
        inv_data[self.DIMS] = problem.cone_dims

        constr_map = problem.constr_map
        inv_data[self.EQ_CONSTR] = constr_map[Zero]
        inv_data[self.NEQ_CONSTR] = constr_map[NonNeg]

        c, d, A, b = problem.apply_parameters()
        data[s.C] = c
        inv_data[s.OFFSET] = d
        data[s.A] = sp.csc_matrix(A)
        if data[s.A].shape[0] == 0:
            data[s.A] = None
        data[s.B] = -b.flatten()
        if data[s.B].shape[0] == 0:
            data[s.B] = None
        data["neq"] = problem.cone_dims.zero
        return data, inv_data

    def solve_via_data(
        self, data, warm_start: bool, verbose: bool, solver_opts, solver_cache=None
    ):
        import culpy

        solution = culpy.solve(
            data[s.A].shape[0],
            data[s.A].shape[1],
            data[s.A].nnz,
            data['neq'],
            data[s.A].indptr,
            data[s.A].indices,
            data[s.A].data,
            data[s.B],
            data[s.C],
            solver_opts,
        )
        return solution

    def invert(self, solution, inverse_data):
        """Returns solution to original problem, given inverse_data."""
        status = self.STATUS_MAP[solution["status"]]

        # Timing data
        attr = {}
        attr[s.SOLVE_TIME] = solution["solve_time"]
        attr[s.SETUP_TIME] = solution["setup_time"]
        attr[s.NUM_ITERS] = solution["num_iters"]

        if status in s.SOLUTION_PRESENT:
            primal_val = solution["pcost"]
            opt_val = primal_val + inverse_data[s.OFFSET]
            primal_vars = {
                inverse_data[self.VAR_ID]: intf.DEFAULT_INTF.const_to_matrix(
                    solution["primal_vars"]
                )
            }
            dual_vars = utilities.get_dual_values(
                solution["dual_vars"],
                utilities.extract_dual_value,
                inverse_data[self.EQ_CONSTR] + inverse_data[self.NEQ_CONSTR],
            )
            return Solution(status, opt_val, primal_vars, dual_vars, attr)
        else:
            return failure_solution(status, attr)
