#!/usr/bin/env python
"""
Google Drive Trash Cleaner

Clears the Google Drive trash folder periodically.
Runs every 10 minutes by default.

Usage:
    # Using OAuth (browser-based login):
    python scripts/clear_drive_trash.py --auth
    python scripts/clear_drive_trash.py

    # Run once:
    python scripts/clear_drive_trash.py --once

    # Run continuously every 10 minutes:
    python scripts/clear_drive_trash.py

    # Run with custom interval (in minutes):
    python scripts/clear_drive_trash.py --interval 5

Setup:
    1. Go to https://console.cloud.google.com/apis/library/drive.googleapis.com
    2. Click "Enable" to enable Google Drive API
    3. Go to APIs & Services > Credentials
    4. Create OAuth 2.0 Client ID (Desktop application)
    5. Download JSON and save as 'scripts/credentials.json'
    6. Run: python scripts/clear_drive_trash.py --auth

Requirements:
    pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
"""

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Scopes required for trash operations
SCOPES = ["https://www.googleapis.com/auth/drive"]

# Token file location
TOKEN_FILE = Path(__file__).parent / "drive_token.json"
CREDENTIALS_FILE = Path(__file__).parent / "credentials.json"


def get_drive_service():
    """Authenticate and return Google Drive service."""
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError:
        logger.error(
            "Missing required packages. Install with:\n"
            "pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib"
        )
        raise

    creds = None

    # Load existing token
    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)

    # Refresh or get new credentials
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            logger.info("Refreshing expired credentials...")
            creds.refresh(Request())
        else:
            if not CREDENTIALS_FILE.exists():
                logger.error(
                    f"\nCredentials file not found: {CREDENTIALS_FILE}\n\n"
                    "=== SETUP INSTRUCTIONS ===\n"
                    "1. Go to: https://console.cloud.google.com/apis/library/drive.googleapis.com\n"
                    "2. Click 'Enable' to enable Google Drive API\n"
                    "3. Go to: https://console.cloud.google.com/apis/credentials\n"
                    "4. Click '+ CREATE CREDENTIALS' > 'OAuth client ID'\n"
                    "5. If prompted, configure OAuth consent screen:\n"
                    "   - Choose 'External'\n"
                    "   - Add your email as a test user\n"
                    "6. Application type: 'Desktop app'\n"
                    "7. Download the JSON file\n"
                    "8. Save it as: scripts/credentials.json\n"
                    "9. Run: python scripts/clear_drive_trash.py --auth\n"
                )
                raise FileNotFoundError(f"Missing {CREDENTIALS_FILE}")

            logger.info("Starting OAuth flow - a browser window will open...")
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_FILE), SCOPES)
            creds = flow.run_local_server(port=0)

        # Save credentials for next run
        with open(TOKEN_FILE, "w") as token:
            token.write(creds.to_json())
        logger.info(f"Credentials saved to {TOKEN_FILE}")

    return build("drive", "v3", credentials=creds)


def empty_trash(service):
    """Empty the Google Drive trash."""
    try:
        # Get trash item count first (for logging)
        results = (
            service.files()
            .list(
                q="trashed=true",
                spaces="drive",
                fields="files(id, name, size)",
                pageSize=1000,
            )
            .execute()
        )

        files = results.get("files", [])
        total_size = sum(int(f.get("size", 0)) for f in files)

        if not files:
            logger.info("Trash is already empty")
            return 0

        logger.info(f"Found {len(files)} items in trash ({total_size / 1024 / 1024:.2f} MB)")

        # Empty the trash
        service.files().emptyTrash().execute()

        logger.info(f"Successfully emptied trash: {len(files)} items deleted")
        return len(files)

    except Exception as e:
        logger.error(f"Failed to empty trash: {e}")
        raise


def run_once(service):
    """Run trash cleanup once."""
    logger.info("=" * 50)
    logger.info(f"Running trash cleanup at {datetime.now()}")
    logger.info("=" * 50)

    try:
        count = empty_trash(service)
        return count
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        return -1


def run_continuous(service, interval_minutes: int = 10):
    """Run trash cleanup continuously at specified interval."""
    logger.info(f"Starting continuous trash cleanup every {interval_minutes} minutes")
    logger.info("Press Ctrl+C to stop")

    iteration = 0
    while True:
        iteration += 1
        logger.info(f"\n--- Iteration {iteration} ---")

        try:
            run_once(service)
        except Exception as e:
            logger.error(f"Error in iteration {iteration}: {e}")

        logger.info(f"Sleeping for {interval_minutes} minutes...")
        try:
            time.sleep(interval_minutes * 60)
        except KeyboardInterrupt:
            logger.info("\nStopped by user")
            break


def main():
    parser = argparse.ArgumentParser(
        description="Google Drive Trash Cleaner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Authenticate (first time setup):
    python scripts/clear_drive_trash.py --auth

    # Run once:
    python scripts/clear_drive_trash.py --once

    # Run every 10 minutes (default):
    python scripts/clear_drive_trash.py

    # Run every 5 minutes:
    python scripts/clear_drive_trash.py --interval 5
        """,
    )

    parser.add_argument(
        "--auth",
        action="store_true",
        help="Only authenticate (useful for first-time setup)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run once and exit",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="Interval in minutes between cleanups (default: 10)",
    )

    args = parser.parse_args()

    # Get authenticated service
    logger.info("Authenticating with Google Drive...")
    service = get_drive_service()
    logger.info("Authentication successful")

    if args.auth:
        logger.info("Authentication complete. Exiting.")
        return

    if args.once:
        run_once(service)
    else:
        run_continuous(service, args.interval)


if __name__ == "__main__":
    main()
