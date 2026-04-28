from email import utils
import os
import uuid
import hashlib
from datetime import datetime, timedelta
import re
from flask import request

class ResultDTO():
    def __init__(self, code, message, data=None, success=False):
        self.code = code
        self.message = message
        self.data = data
        self.success = success # 내부 결과 소통용

    def to_response(self):
        response = {
            "code": self.code,
            "message": self.message,
            "data": self.data
        }
        return response, self.code

def generate_uuid() -> str:
    return str(uuid.uuid4())

def generate_hash(len: int) -> str:
    return os.urandom(16).hex()[:len]

def str_to_hash(input_string: str) -> str:
    return hashlib.sha256(input_string.encode()).hexdigest()

def get_request_ip(): 
    user_ip = request.headers.get("X-Forwarded-For", request.remote_addr).split(",")[0].strip()
    return f'{user_ip}'

def check_password(password: str, hashed_password: str, salt: str) -> bool:
    return str_to_hash(password + salt) == hashed_password

# Datetime
def get_current_datetime() -> datetime:
    return datetime.utcnow() + timedelta(hours=9)

def get_current_datetime_str() -> str:
    return (datetime.utcnow() + timedelta(hours=9)).strftime('%Y-%m-%d %H:%M:%S')

def str_to_datetime(date_str: str) -> datetime:
    return datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')

def datetime_to_str(date: datetime) -> str:
    return date.strftime('%Y-%m-%d %H:%M:%S')

def is_minutes_passed(start_time: str, minutes: int) -> bool:
    start_dt = str_to_datetime(start_time)
    target_time = start_dt + timedelta(minutes=minutes)
    now_time = get_current_datetime()
    print(now_time, target_time)
    return now_time > target_time

def get_future_timestamp(days: int = 0, hours: int = 0, minutes: int = 0, seconds: int = 0) -> datetime:
    future_datetime = get_current_datetime() + timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)
    return future_datetime

# regex
class RegexResultDTO():
    def __init__(self, success: bool, detail: str):
        self.success = success
        self.detail = detail

def is_valid_email(email: str) -> RegexResultDTO:
    # Step 1: Check email format
    if not re.fullmatch(r'^[\w\.-]+@[\w\.-]+\.\w+$', email):
        return RegexResultDTO(
            success=False,
            detail="이메일 형식이 유효하지 않습니다."
        )
    
    return RegexResultDTO(success=True, detail="올바른 이메일입니다.")

def is_valid_id(id: str) -> RegexResultDTO:
    # Step 1: Check only alphanumeric
    if not re.fullmatch(r'[a-zA-Z0-9]+', id):
        return RegexResultDTO(
            success=False,
            detail="아이디는 영숫자만 사용할 수 있습니다."
        )

    # Step 2: Check length between 2 and 20
    if not re.fullmatch(r'.{2,20}', id):
        return RegexResultDTO(
            success=False,
            detail="아이디는 2~20자 사이여야 합니다."
        )

    return RegexResultDTO(success=True, detail="올바른 아이디입니다.")

def is_valid_password(password: str) -> RegexResultDTO:
    # Step 1: Check only alphanumeric
    if not re.fullmatch(r'[a-zA-Z0-9!@#$%^&*()_+]+', password):
        return RegexResultDTO(
            success=False,
            detail="비밀번호는 영숫자와 특수문자만 사용할 수 있습니다."
        )

    # Step 2: Check length between 8 and 20
    if not re.fullmatch(r'.{8,20}', password):
        return RegexResultDTO(
            success=False,
            detail="비밀번호는 8~20자 사이여야 합니다."
        )

    return RegexResultDTO(success=True, detail="올바른 비밀번호입니다.")


