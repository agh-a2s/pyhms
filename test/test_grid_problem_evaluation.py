from pyhms.utils.visualisation.grid import Grid2DProblemEvaluation

from .config import SQUARE_BOUNDS, SQUARE_PROBLEM


def test_grid_problem_evaluation():
    N = 2
    granularity = (SQUARE_BOUNDS[0][1] - SQUARE_BOUNDS[0][0]) // N
    grid = Grid2DProblemEvaluation(SQUARE_PROBLEM, SQUARE_BOUNDS, granularity=granularity)
    grid.evaluate()
    assert grid.imshow_view.shape == (N + 1, N + 1)
    assert grid.imshow_view[N // 2, N // 2] == 0.0
