import pickle
import os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from typing import List

# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']

class GoogleSheetReader:
    '''A class to abstract over reading google sheet(s)

    Based on https://developers.google.com/sheets/api/quickstart/python

    1. Enable the google sheets api and obtain a json file with your credentials
    2. Run the app and sign in to your google account to allow access to your
       google sheets
    '''

    MISSING_CREDENTIALS_ERROR = '''
    Could not find file credentials file {}. You can specify a different path
    when creating an instance of this class.

    This file should contain information obtained from enabling the google
    sheets api in the google developers console https://console.developers.google.com/
    '''


    def __init__(self, credentials_path='./credentials.json',
            token_path='./token.pickle'):
        self._creds = None
        if os.path.exists(token_path):
            with open(token_path, 'rb') as token:
                self._creds = pickle.load(token)
        # If there are no (valid) credentials available, let the user log in.
        if not self._creds or not self._creds.valid:
            if self._creds and self._creds.expired and self._creds.refresh_token:
                self._creds.refresh(Request())
            else:
                if not os.path.exists(credentials_path):
                    raise FileNotFoundError(
                        self.MISSING_CREDENTIALS_ERROR.format(credentials_path))
                flow = InstalledAppFlow.from_client_secrets_file(
                    credentials_path, SCOPES)
                self._creds = flow.run_local_server()
            # Save the credentials for the next run
            with open(token_path, 'wb') as token:
                pickle.dump(self._creds, token)
        self._service = build('sheets', 'v4', credentials=self._creds)

    def read_sheet(self, sheet_id: str, sheet_range: str) -> List[List[str]]:
        '''
        Returns the given range from the given spreadsheet as a list of lists
        '''
        sheet = self._service.spreadsheets()
        result = sheet.values().get(
            spreadsheetId=sheet_id, range=sheet_range).execute()
        return result.get('values', [])
