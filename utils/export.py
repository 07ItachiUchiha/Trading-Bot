import os
import pandas as pd
import logging
from pathlib import Path
import datetime
import json

# Configure logging
logger = logging.getLogger('export')

# Create exports directory if not exists
EXPORT_DIR = os.path.join(Path(__file__).parent.parent, "exports")
os.makedirs(EXPORT_DIR, exist_ok=True)

def export_to_excel(trades_df):
    """Save trade history as an Excel file in exports/."""
    try:
        # Format data for export - create an explicit copy to avoid SettingWithCopyWarning
        export_df = trades_df.copy(deep=True)
        
        # Convert JSON targets to string
        if 'targets' in export_df.columns:
            export_df['targets'] = export_df['targets'].apply(lambda x: ', '.join([f"{t:.2f}" for t in x]) if isinstance(x, list) else x)
        
        # Generate filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"trade_history_{timestamp}.xlsx"
        file_path = os.path.join(EXPORT_DIR, filename)
        
        # Write to Excel
        export_df.to_excel(file_path, index=False)
        
        logger.info(f"Trade history exported to Excel: {file_path}")
        return file_path
        
    except Exception as e:
        logger.error(f"Error exporting to Excel: {str(e)}")
        return None

def export_to_google_sheets(trades_df):
    """
    Placeholder for Google Sheets export.
    For now just saves a CSV - would need Sheets API credentials for the real thing.
    """
    try:
        # Format data for export - create an explicit copy to avoid SettingWithCopyWarning
        export_df = trades_df.copy(deep=True)
        
        # Convert JSON targets to string
        if 'targets' in export_df.columns:
            export_df['targets'] = export_df['targets'].apply(lambda x: ', '.join([f"{t:.2f}" for t in x]) if isinstance(x, list) else x)
        
        # Generate filename with timestamp for saving locally first
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"trade_history_{timestamp}.csv"
        file_path = os.path.join(EXPORT_DIR, filename)
        
        # Write to CSV
        export_df.to_csv(file_path, index=False)
        
        # TODO: Implement Google Sheets API integration with OAuth2 credentials
        logger.info(f"Trade history exported to CSV: {file_path}")
        logger.warning("Google Sheets integration not yet implemented. Use CSV export for now.")
        
    except Exception as e:
        logger.error(f"Error exporting to Google Sheets: {str(e)}")
        return None
