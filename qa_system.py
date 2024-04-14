import os
from dotenv import load_dotenv
import pickle 
import shutil
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.vectorstores.faiss import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatYandexGPT
from langchain_community.embeddings.yandex import YandexGPTEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import ConfigurableFieldSpec

class QASystem():
    def __init__(self, docs_store = "./db_files", index_store = "./faiss_db"):
        # initialize parameters and meaning parts
        print("Инициализация параметров!")
        load_dotenv()
        self.iam_token, self.api_key, self.folder_id = self.get_connect_info()
        self.index_store = index_store
        self.docs_store = docs_store

        self.embedder = YandexGPTEmbeddings(iam_token=self.iam_token,
                                            api_key=self.api_key,
                                            model_uri=f"emb://{self.folder_id}/text-search-query/latest",
                                            sleep_interval = 0.1)
        
        self.reader = ChatYandexGPT(iam_token=self.iam_token,
                                    api_key=self.api_key,
                                    model_uri=f"gpt://{self.folder_id}/yandexgpt/latest",
                                    )

        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, 
                                                            chunk_overlap=100, 
                                                            separators=['\n\n', '\n'])

        # load or create database
        print("Загрузка БД!")
        if os.path.exists(self.index_store):
            self.doc_db = FAISS.load_local(self.index_store, self.embedder, allow_dangerous_deserialization=True)
        else:
            self.doc_db = self.split_documents()

        print("Загрузка истории диалогов!")
        if os.path.isfile("./dialogs_history.pkl"):
            with open('dialogs_history.pkl', 'rb') as f:
                self.store = pickle.load(f)
        else:
            self.store = {}

        # set retriever
        self.retriever = self.doc_db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

        # set prompt with dialog history
        print("Создаем шаблон промпта с историей диалога!")
        template = self.get_prompt_template()
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", template),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ]
        )

        # RetrievalQA
        self.qa_chain_with_history = RunnableWithMessageHistory(
            self.prompt | self.reader,
            self.get_session_history,
            input_messages_key="question",
            history_messages_key="history",
            history_factory_config=[
                ConfigurableFieldSpec(
                    id="user_id",
                    annotation=str,
                    name="User ID",
                    description="Unique identifier for the user.",
                    default="",
                    is_shared=True,
                ),
                ConfigurableFieldSpec(
                    id="conversation_id",
                    annotation=str,
                    name="Conversation ID",
                    description="Unique identifier for the conversation.",
                    default="",
                    is_shared=True,
                ),
            ],
        )

    def get_connect_info(self):
        return os.getenv("IAM_TOKEN"), os.getenv("API_KEY"), os.getenv("FOLDER_ID")
    
    def split_documents(self):
        print("Разбиваем документы на чанки!")
        docs = DirectoryLoader(self.docs_store, glob = "**/*.txt", show_progress = True,
                                    loader_kwargs = {'autodetect_encoding': True}).load()
        splits = self.text_splitter.split_documents(docs)
        doc_db = FAISS.from_documents(documents=splits, embedding=self.embedder)
        doc_db.save_local(folder_path=self.index_store)
        return doc_db

    def get_prompt_template(self):
        template = "Ты очень полезный чатбот, тебя зовут YandexGPT. Можешь общаться на разные темы. \
        При ответе на вопросы будь краток, используй 30 слов или меньше."
        return template

    def add_new_embedding(self, doc_paths_list):
        new_docs = []
        for doc_path in doc_paths_list:
            new_docs.append(TextLoader(doc_path).load()[0])
            shutil.copy(doc_path, self.docs_store)

        splits = self.text_splitter.split_documents(new_docs)
        new_docs = FAISS.from_documents(documents=splits, embedding=self.embedder)
        
        if os.path.exists(self.index_store):
            local_db = FAISS.load_local(self.index_store, self.embedder, allow_dangerous_deserialization=True)
            local_db.merge_from(new_docs)

        self.doc_db = local_db
        local_db.save_local(self.index_store)

    def get_session_history(self, user_id, conversation_id):
        if (user_id, conversation_id) not in self.store:
            self.store[(user_id, conversation_id)] = ChatMessageHistory()
        return self.store[(user_id, conversation_id)]

    def save_dialogs_history(self):
        with open('./dialogs_history.pkl', 'wb') as f:
            pickle.dump(self.store, f)

    def run_pipeline(self, questions, user_id, conversation_id):
        predictions = []
        for question in questions:
            predictions.append(self.qa_chain_with_history.invoke({"question": question}, 
                                                config={"configurable": {"user_id": user_id, "conversation_id": conversation_id}}))
        self.save_dialogs_history()
        return predictions