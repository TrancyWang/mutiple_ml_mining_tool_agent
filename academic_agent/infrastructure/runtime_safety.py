"""Runtime safeguards for native numerical and tokenizer libraries.

The academic agent runs model inference and scikit-learn inside a Qt process.
Those libraries eventually call native BLAS/OpenMP code, which is not always
safe after a fork or when several thread pools compete for the same process.
Keep the defaults conservative and predictable before importing those stacks.
"""

from __future__ import annotations

import os


_SAFE_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "KMP_INIT_AT_FORK": "FALSE",
}


def configure_numeric_runtime() -> None:
    """Set process-wide defaults before NumPy, tokenizers or sklearn load.

    Numerical thread pools are deliberately forced to one worker for this
    desktop application. A deployment can opt out for benchmarking with
    ``ACADEMIC_AGENT_ALLOW_PARALLEL_NUMERICS=1``. The tokenizer setting is
    always disabled because a tokenizer pool inherited across a fork is the
    source of the warning and possible deadlock seen in the app.
    """

    if os.getenv("ACADEMIC_AGENT_ALLOW_PARALLEL_NUMERICS", "0") != "1":
        for name, value in _SAFE_THREAD_ENV.items():
            os.environ[name] = value
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def configure_torch_threads() -> None:
    """Limit already-loaded PyTorch pools when the model worker starts."""

    try:
        import torch
    except ImportError:
        return

    try:
        torch.set_num_threads(1)
    except RuntimeError:
        # PyTorch may have already initialized its pool in an embedding call.
        pass
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # This setter is only legal before the first parallel operation.
        pass


configure_numeric_runtime()
