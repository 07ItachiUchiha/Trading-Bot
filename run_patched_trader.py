"""
Patched trading bot script that fixes the numpy.NaN import issue in pandas_ta.
This script patches numpy to add the uppercase NaN constant before importing pandas_ta.
"""
import numpy as np

# Add this patch to fix the NaN import issue in pandas_ta
# This makes numpy.NaN available for pandas_ta modules that import it directly
if not hasattr(np, 'NaN'):
    np.NaN = np.nan  # Use lowercase nan which is always available

# Now that numpy is patched, we can safely import the main script
import main  # This will run the main script with the patched numpy

# Alternatively, you can put your main code here instead of importing main.py