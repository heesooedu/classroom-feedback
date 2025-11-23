import os
import yaml
import json
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.templating import Jinja2Templates
from sqlmodel import Session, select
from dotenv import load_dotenv
import google.generativeai as genai

from database import create_db_and_tables, get_session, engine
from models import (Course, Classroom, Student, Submission, 
                    SetupRequest, LoginRequest, SubmitRequest, 
                    ProgressUpdateRequest, ActivateClassRequest)

# --- 설정 ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# 모델 설정: Gemini 2.5 Flash (정확) / 2.5 flash lite RPM이 높음, 채점이 아주 정확하진 않음
target_model_name = 'gemini-2.5-flash'
model = genai.GenerativeModel(target_model_name, generation_config={"response_mime_type": "application/json"})

SYSTEM_PROMPT = """
당신은 학교 선생님을 돕는 유능한 AI 보조교사입니다.
학생의 코드를 채점하고 피드백을 줄 때 다음 원칙을 반드시 지키세요:
0. 모든 해설은 한국어로 해주세요.
1. 정답 코드를 바로 알려주지 마세요. 학생이 스스로 고칠 수 있도록 힌트(Scaffolding)를 제공하세요.
2. 칭찬을 먼저 하고, 고쳐야 할 점을 부드럽게 이야기하세요.
3. 점수는 코드의 정확성과 문제 요구사항 충족 여부에 따라 0~100점 사이 정수로 매기세요.
4. 문법 에러가 있다면 어디가 틀렸는지 구체적으로 지적하세요.
5. 학생들은 예외처리(try-except) 같이 어려운 문법은 배우지 않았습니다. 정말 기초 문법 수준에서 대답해주세요.
"""

# 채점 대기열 (비동기 큐)
submission_queue = asyncio.Queue()

with open("problems.yaml", "r", encoding="utf-8") as f:
    raw_data = yaml.safe_load(f)
    if isinstance(raw_data, list):
        PROBLEMS_BY_COURSE = {"기본과목": raw_data}
    else:
        PROBLEMS_BY_COURSE = raw_data

    PROBLEMS_DICT = {}
    CHAPTERS_BY_COURSE = {}

    for course_name, p_list in PROBLEMS_BY_COURSE.items():
        chapters = sorted(list(set(p.get('chapter', 'Unknown') for p in p_list)))
        CHAPTERS_BY_COURSE[course_name] = chapters
        for p in p_list:
            p['course_name'] = course_name
            PROBLEMS_DICT[p['id']] = p

# --- 백그라운드 워커 (순차 채점) ---
async def process_submission_queue():
    print("🚀 채점 워커(Worker) 가동됨 (6.5초 간격)")
    while True:
        # 큐에서 작업 가져오기
        submission_id, problem_info, code = await submission_queue.get()
        
        try:
            with Session(engine) as session:
                submission = session.get(Submission, submission_id)
                if not submission:
                    submission_queue.task_done()
                    continue

                print(f"🤖 AI 채점 시작: ID {submission_id} ...")
                
                prompt = f"""
                {SYSTEM_PROMPT}
                [Problem] {problem_info['title']}
                [Desc] {problem_info['description']}
                [Criteria] {problem_info['ai_prompt']}
                [Code]
                {code}
                Return SINGLE JSON: "score"(int), "feedback"(str)
                """
                
                # 비동기적으로 Gemini 호출
                response = await asyncio.to_thread(model.generate_content, prompt)
                text_res = response.text.strip()
                if text_res.startswith("```"):
                    text_res = text_res.replace("```json", "").replace("```", "")
                
                res_json = json.loads(text_res)
                if isinstance(res_json, list): 
                    res_json = res_json[0] if res_json else {}

                submission.score = res_json.get("score", 0)
                submission.ai_feedback = res_json.get("feedback", "피드백 생성 실패")
                submission.status = "completed"
                
                session.add(submission)
                session.commit()
                print(f"✅ 채점 완료: ID {submission_id}")

        except Exception as e:
            print(f"❌ 채점 오류: {e}")
            with Session(engine) as session:
                submission = session.get(Submission, submission_id)
                if submission:
                    submission.score = 0
                    submission.ai_feedback = "서버 사용량이 많아 채점에 실패했습니다. 잠시 후 다시 시도해주세요."
                    submission.status = "completed"
                    session.add(submission)
                    session.commit()
        
        finally:
            submission_queue.task_done()
            # [속도 조절] 10 RPM 제한 준수 (6초 이상 대기) / 2.5 flash lite 4.5초
            await asyncio.sleep(6.5)

@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db_and_tables()
    asyncio.create_task(process_submission_queue())
    yield

app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="templates")

# --- 기본 페이지 ---
@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/admin")
async def read_admin(request: Request):
    return templates.TemplateResponse("admin.html", {"request": request})

# --- 시스템 API ---
@app.get("/api/system/info")
async def get_system_info(session: Session = Depends(get_session)):
    courses = session.exec(select(Course)).all()
    classes = session.exec(select(Classroom)).all()
    
    course_map = {c.id: c.name for c in courses}
    class_list = []
    for cls in classes:
        c_name = course_map.get(cls.course_id, "Unknown")
        class_list.append({
            "id": cls.id, "name": cls.name, "course_id": cls.course_id, "course_name": c_name,
            "display_name": f"[{c_name}] {cls.name}", "active_chapter": cls.active_chapter, "is_active": cls.is_active
        })

    return {"initialized": len(courses) > 0, "courses": courses, "classes": class_list, 
            "available_courses_in_yaml": list(PROBLEMS_BY_COURSE.keys()), "chapters_by_course": CHAPTERS_BY_COURSE}

@app.post("/api/system/setup")
async def setup_system(req: SetupRequest, session: Session = Depends(get_session)):
    course = Course(name=req.course_name)
    session.add(course); session.commit(); session.refresh(course)
    chapters = CHAPTERS_BY_COURSE.get(req.course_name, [])
    def_chap = chapters[0] if chapters else ""
    for name in req.class_names:
        session.add(Classroom(course_id=course.id, name=name, active_chapter=def_chap))
    session.commit()
    return {"status": "success"}

# --- 학생 API ---
@app.get("/api/student/active_classes")
async def get_active_classes(session: Session = Depends(get_session)):
    active_classes = session.exec(select(Classroom).where(Classroom.is_active == True)).all()
    result = []
    for cls in active_classes:
        course = session.get(Course, cls.course_id)
        result.append({"id": cls.id, "display_name": f"[{course.name}] {cls.name}"})
    return result

@app.post("/api/login")
async def login(req: LoginRequest, session: Session = Depends(get_session)):
    student = session.exec(select(Student).where(Student.student_number == req.student_number)).first()
    if not student:
        student = Student(classroom_id=req.classroom_id, student_number=req.student_number, name=req.name)
        session.add(student)
    else:
        student.classroom_id = req.classroom_id; student.name = req.name; session.add(student)
    session.commit(); session.refresh(student)
    classroom = session.get(Classroom, student.classroom_id)
    course = session.get(Course, classroom.course_id)
    return {"id": student.id, "name": student.name, "class_name": classroom.name, "course_name": course.name, "classroom_id": classroom.id}

@app.get("/api/problems")
async def get_problems(student_id: int, session: Session = Depends(get_session)):
    student = session.get(Student, student_id)
    if not student: raise HTTPException(404)
    classroom = session.get(Classroom, student.classroom_id)
    course = session.get(Course, classroom.course_id)
    
    course_problems = PROBLEMS_BY_COURSE.get(course.name, [])
    target_problems = [p for p in course_problems if p['chapter'] == classroom.active_chapter]
    p_ids = [p['id'] for p in target_problems]
    
    submissions = session.exec(select(Submission).where(Submission.student_id == student_id).where(Submission.problem_id.in_(p_ids)).order_by(Submission.created_at.desc())).all()
    sub_map = {}
    for sub in submissions:
        if sub.problem_id not in sub_map: sub_map[sub.problem_id] = sub
            
    enriched = []
    for p in target_problems:
        pc = p.copy()
        if p['id'] in sub_map:
            s = sub_map[p['id']]
            pc.update({'has_submission': True, 'last_code': s.code_answer, 'last_score': s.score, 'last_feedback': s.ai_feedback, 'status': s.status})
        else:
            pc['has_submission'] = False
        enriched.append(pc)
    return {"active_chapter": classroom.active_chapter, "problems": enriched}

@app.post("/api/submit")
async def submit(req: SubmitRequest, session: Session = Depends(get_session)):
    problem = PROBLEMS_DICT.get(req.problem_id)
    if not problem: raise HTTPException(404)

    # 1. 'grading' 상태로 저장
    submission = Submission(
        student_id=req.student_id, problem_id=req.problem_id, 
        code_answer=req.code_answer, status="grading", 
        ai_feedback="채점 대기열에 등록되었습니다. 잠시만 기다려주세요..."
    )
    session.add(submission); session.commit(); session.refresh(submission)

    # 2. 큐에 추가
    await submission_queue.put((submission.id, problem, req.code_answer))

    return submission

# [중요] 상태 확인 폴링 API
@app.get("/api/check_submission/{submission_id}")
async def check_submission(submission_id: int, session: Session = Depends(get_session)):
    sub = session.get(Submission, submission_id)
    if not sub: raise HTTPException(404)
    session.refresh(sub) # DB에서 최신 정보 강제 갱신
    return sub

# --- 교사 API (동일) ---
@app.post("/api/admin/activate")
async def activate_class(req: ActivateClassRequest, session: Session = Depends(get_session)):
    all_classes = session.exec(select(Classroom)).all()
    for cls in all_classes: cls.is_active = False; session.add(cls)
    target = session.get(Classroom, req.classroom_id)
    if target: target.is_active = True; session.add(target)
    session.commit()
    return {"status": "activated"}

@app.post("/api/admin/progress")
async def update_progress(req: ProgressUpdateRequest, session: Session = Depends(get_session)):
    classroom = session.get(Classroom, req.classroom_id)
    if not classroom: raise HTTPException(404)
    classroom.active_chapter = req.active_chapter; session.add(classroom); session.commit()
    return {"status": "updated"}

@app.get("/api/status")
async def get_status(classroom_id: int, session: Session = Depends(get_session)):
    classroom = session.get(Classroom, classroom_id)
    if not classroom: return {"students": [], "problems": []}
    course = session.get(Course, classroom.course_id)
    course_problems = PROBLEMS_BY_COURSE.get(course.name, [])
    target_problems = [p for p in course_problems if p['chapter'] == classroom.active_chapter]
    p_ids = [p['id'] for p in target_problems]
    students = session.exec(select(Student).where(Student.classroom_id == classroom_id)).all()
    s_ids = [s.id for s in students]
    submissions = session.exec(select(Submission).where(Submission.student_id.in_(s_ids)).where(Submission.problem_id.in_(p_ids)).order_by(Submission.created_at.desc())).all()
    sub_map = {}
    for sub in submissions:
        if sub.student_id not in sub_map: sub_map[sub.student_id] = {}
        if sub.problem_id not in sub_map[sub.student_id]: sub_map[sub.student_id][sub.problem_id] = sub
    result = []
    for s in students:
        row = {"info": f"{s.student_number} {s.name}", "problems": {}}
        for p in target_problems:
            item = sub_map.get(s.id, {}).get(p['id'])
            row["problems"][p['id']] = {
                "id": item.id if item else None, "status": item.status if item else "none",
                "score": item.score if item else 0, "feedback": item.ai_feedback if item else "",
                "code": item.code_answer if item else ""
            }
        result.append(row)
    return {"students": result, "problems": target_problems, "chapter": classroom.active_chapter}