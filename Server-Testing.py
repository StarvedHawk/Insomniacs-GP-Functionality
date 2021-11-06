import json
import sys

import requests
from requests.structures import CaseInsensitiveDict

STD_ID = sys.argv[1]
Exam = sys.argv[2]
CurrentTime = "10:58:32"
AI_Message = "No Face Detected"
AI_Danger_Level = 30

url = "http://127.0.0.1:8000/api/TimeLine/"

headers = CaseInsensitiveDict()
headers["Accept"] = "application/json"
headers["Content-Type"] = "application/json"

data = {}
data['student'] = STD_ID
data['CurrentExam'] = Exam
data['AItimeStamp'] = CurrentTime
data['AItextMessage'] = AI_Message
data['AIdangerLevel'] = AI_Danger_Level


try:
    resp = requests.post(url, headers=headers, data=json.dumps(data))
    print(resp.status_code)
except Exception:
    print("Connection Error!")