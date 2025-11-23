from typing import Optional, List
from datetime import datetime
from sqlmodel import Field, SQLModel

# ------------------------------------------------
# 1. 데이터베이스 테이블 모델 (Table Models)
# ------------------------------------------------

class Course(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str

class Classroom(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    course_id: int = Field(foreign_key="course.id")
    name: str
    active_chapter: str = Field(default="")
    is_active: bool = Field(default=False) # 현재 수업 진행 중 여부

class Student(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    classroom_id: int = Field(foreign_key="classroom.id")
    student_number: str
    name: str
    created_at: datetime = Field(default_factory=datetime.now)

class Submission(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    student_id: int = Field(foreign_key="student.id")
    problem_id: int
    code_answer: str
    ai_feedback: Optional[str] = None
    score: Optional[int] = None
    status: str = Field(default="grading")
    created_at: datetime = Field(default_factory=datetime.now)

# ------------------------------------------------
# 2. API 요청 데이터 모델 (Request DTOs)
# ------------------------------------------------

class SetupRequest(SQLModel):
    course_name: str
    class_names: List[str]

class LoginRequest(SQLModel):
    classroom_id: int
    student_number: str
    name: str

class SubmitRequest(SQLModel):
    student_id: int
    problem_id: int
    code_answer: str

class ProgressUpdateRequest(SQLModel):
    classroom_id: int
    active_chapter: str

class ActivateClassRequest(SQLModel):
    classroom_id: int