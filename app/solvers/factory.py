"""
Solver Factory.

Creates solver instances based on configuration.
Enables easy switching between optimization solvers.
"""

from typing import Optional

from app.core.config import Settings
from app.solvers.interfaces.solver import ISolver
from app.solvers.cvxpy_solver import CvxpySolver
from app.solvers.scipy_solver import ScipySolver


class SolverFactory:
    """
    Factory for creating optimization solver instances.
    
    Supports:
    - CVXPY (default): Convex optimization with multiple backends
    - SciPy: Alternative using scipy.optimize
    
    Example:
        # From settings
        settings = get_settings()
        solver = SolverFactory.create_from_settings(settings)
        
        # Direct creation
        solver = SolverFactory.create("cvxpy")
    """

    SUPPORTED_SOLVERS = ["cvxpy", "scipy"]

    @classmethod
    def create(
        cls,
        solver_name: str,
        **kwargs,
    ) -> ISolver:
        """
        Create a solver instance.
        
        Args:
            solver_name: Solver name ('cvxpy', 'scipy')
            **kwargs: Additional solver-specific arguments
            
        Returns:
            ISolver instance
            
        Raises:
            ValueError: If solver_name is not supported
        """
        solver_name = solver_name.lower()
        
        if solver_name not in cls.SUPPORTED_SOLVERS:
            raise ValueError(
                f"Unsupported solver: {solver_name}. "
                f"Supported solvers: {cls.SUPPORTED_SOLVERS}"
            )
        
        if solver_name == "cvxpy":
            return cls._create_cvxpy(**kwargs)
        elif solver_name == "scipy":
            return cls._create_scipy(**kwargs)
        else:
            raise ValueError(f"Unsupported solver: {solver_name}")

    @classmethod
    def create_from_settings(cls, settings: Settings) -> ISolver:
        """
        Create a solver from application settings.
        
        Args:
            settings: Application settings instance
            
        Returns:
            ISolver instance configured from settings
        """
        solver_name = settings.solver
        return cls.create(solver_name)

    @classmethod
    def _create_cvxpy(
        cls,
        solver_backend: Optional[str] = None,
        **kwargs,
    ) -> CvxpySolver:
        """Create CVXPY solver instance."""
        return CvxpySolver(solver_backend=solver_backend)

    @classmethod
    def _create_scipy(
        cls,
        method: str = "SLSQP",
        max_iterations: int = 1000,
        **kwargs,
    ) -> ScipySolver:
        """Create SciPy solver instance."""
        return ScipySolver(method=method, max_iterations=max_iterations)

    @classmethod
    def get_available_solvers(cls) -> list[str]:
        """Get list of available (installed) solvers."""
        available = []
        
        # Check CVXPY
        try:
            import cvxpy
            available.append("cvxpy")
        except ImportError:
            pass
        
        # Check SciPy (should always be available with numpy)
        try:
            from scipy.optimize import minimize
            available.append("scipy")
        except ImportError:
            pass
        
        return available

    @classmethod
    def get_default_solver(cls) -> str:
        """Get the default solver (first available)."""
        available = cls.get_available_solvers()
        if "cvxpy" in available:
            return "cvxpy"
        elif "scipy" in available:
            return "scipy"
        else:
            raise RuntimeError("No optimization solvers available")


def create_solver(settings: Optional[Settings] = None) -> ISolver:
    """
    Convenience function to create a solver.
    
    Args:
        settings: Optional settings instance (uses get_settings() if not provided)
        
    Returns:
        Configured ISolver instance
    """
    if settings is None:
        from app.core.config import get_settings
        settings = get_settings()
    
    return SolverFactory.create_from_settings(settings)

