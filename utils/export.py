import os
import pandas as pd
import logging
from pathlib import Path
import datetime
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('export')

# Create exports directory if not exists
EXPORT_DIR = os.path.join(Path(__file__).parent.parent, "exports")
os.makedirs(EXPORT_DIR, exist_ok=True)

def export_to_excel(trades_df):
    """
    Export trade history to Excel file
    
    Args:
        trades_df (pandas.DataFrame): DataFrame containing trade history
        
    Returns:
        str: Path to exported Excel file
    """
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
    Export trade history to Google Sheets
    Note: This is a placeholder function that demonstrates how it would work.
    In a real implementation, this would use the Google Sheets API.
    
    Args:
        trades_df (pandas.DataFrame): DataFrame containing trade history
        
    Returns:
        str: URL to Google Sheet (mock)
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
        
        # In a real implementation, this would use Google Sheets API
        # For demonstration, we just log the message and return a mock URL
        logger.info(f"Trade history exported to CSV (ready for Google Sheets): {file_path}")
        logger.info("NOTE: Google Sheets integration requires OAuth2 credentials and Google Sheets API setup")
        
        # Mock URL for demonstration
        sheet_url = "https://docs.google.com/spreadsheets/d/example-sheet-id"
        
        return sheet_url
        
    except Exception as e:
        logger.error(f"Error exporting to Google Sheets: {str(e)}")
        return None