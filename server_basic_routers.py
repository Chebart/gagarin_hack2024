from typing import List
from fastapi import APIRouter, Query
from server_models import qa_model

router = APIRouter()

@router.post("/add-new-txt-files-to-db", tags=["basic"])
async def add_new_txt_files_to_db(doc_paths: List[str] = Query(None)):
    """
    Принимает список путей новых файлов, считывает их,
    разбивает на чанки и добавляет их в БД.
    """
    qa_model.add_new_embedding(doc_paths)
    return "данные успешно добавлены!"

@router.post("/get-dialog-history", tags=["basic"])
async def get_dialog_history(user_id: str = None,
                             conversation_id: str = None):
    """
    Принимает индексы пользователя и беседы, а возвращает
    всю историю соответсвующего диалога
    """
    dialog_history_with_params = dict(qa_model.get_session_history(user_id, conversation_id))["messages"]
    dialog_history = []

    for i in range(0, len(dialog_history_with_params), 2):
        dialog_history.append({"question": dialog_history_with_params[i].content,
                               "answer": dialog_history_with_params[i+1].content})
    return dialog_history

@router.post("/process-question-and-get-answer", tags=["basic"])
async def process_question_and_get_answer(questions: List[str] = Query(None), 
                                          user_id: str = None,
                                          conversation_id: str = None):
    """
    Принимает вопрос и выдает на него ответ
    """
    answers_with_params = qa_model.run_pipeline(questions, user_id, conversation_id)

    answers = []
    for answer in answers_with_params:
        answers.append(answer.content)
    return answers
