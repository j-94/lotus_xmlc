# -*- coding: utf-8 -*-
import os
import pickle
import logging
import time

logger = logging.getLogger(__name__)

def check_or_run(cache_path: str, func, use_cache: bool, force_rerun: bool = False, **kwargs):
    """Checks for a cached result, runs the function if not found or cache disabled/forced.

    Args:
        cache_path: The file path to store/load the cached result (.pkl).
        func: The function to execute if cache is missed.
        use_cache: Boolean flag to enable/disable caching.
        force_rerun: Boolean flag to force execution even if cache exists.
        **kwargs: Arguments to pass to the function `func`.

    Returns:
        The result of the function execution (either loaded from cache or newly computed).
        Returns None if the function fails.
    """
    # Ensure the cache directory exists
    cache_dir = os.path.dirname(cache_path)
    if cache_dir and not os.path.exists(cache_dir):
        try:
            os.makedirs(cache_dir)
            logger.info(f"Created cache directory: {cache_dir}")
        except OSError as e:
            logger.error(f"Failed to create cache directory {cache_dir}: {e}. Caching might fail.")

    if use_cache and not force_rerun and os.path.exists(cache_path):
        try:
            logger.info(f"Loading cached result from: {cache_path}")
            start_load = time.time()
            with open(cache_path, 'rb') as f:
                result = pickle.load(f)
            load_time = time.time() - start_load
            logger.info(f"Loaded cache in {load_time:.2f} seconds.")
            return result
        except (pickle.UnpicklingError, EOFError, FileNotFoundError, Exception) as e:
            logger.warning(f"Failed to load cache from {cache_path}: {e}. Recomputing.")
            # If loading fails, proceed to compute

    # Conditions for running the function:
    # 1. Caching is disabled (use_cache=False)
    # 2. Cache is enabled but file doesn't exist (os.path.exists=False)
    # 3. Cache is enabled but forced rerun (force_rerun=True)
    # 4. Cache loading failed (previous block)

    logger.info(f"Executing function: {func.__name__} (cache {'disabled' if not use_cache else 'missed or forced'})")
    try:
        start_run = time.time()
        result = func(**kwargs)
        run_time = time.time() - start_run
        logger.info(f"Function {func.__name__} executed in {run_time:.2f} seconds.")

        # Save to cache if enabled
        if use_cache:
            try:
                logger.info(f"Saving result to cache: {cache_path}")
                start_save = time.time()
                with open(cache_path, 'wb') as f:
                    pickle.dump(result, f)
                save_time = time.time() - start_save
                logger.info(f"Saved cache in {save_time:.2f} seconds.")
            except (pickle.PicklingError, OSError, Exception) as e:
                logger.error(f"Failed to save cache to {cache_path}: {e}")

        return result

    except Exception as e:
        logger.error(f"Error executing function {func.__name__}: {e}", exc_info=True)
        return None # Return None on function execution error 