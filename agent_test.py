import os
import asyncio
from langchain.chat_models.gigachat import GigaChat
from langchain.schema import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from langchain.document_loaders import TextLoader
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
)

from chromadb.config import Settings
from langchain.vectorstores import Chroma
from langchain_community.embeddings import GigaChatEmbeddings

from langchain.chains import RetrievalQA

from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Optional


# class RelatedSubjects(BaseModel):
#     """
#     class RelatedSubjects(BaseModel):
#         topics: List[str] = Field(
#             description="Дополнительные темы для изучения",
#         )
#     """
#     topics: List[str]
#
#     class Config:
#         schema_extra = {
#             "description": "Дополнительные темы для изучения"
#         }


#  Инициируем LLM
GIGA_TOKEN='Yjg4MTQzMmUtNDAwMS00NDk0LThjOGUtNmU5ZWQ2YzQ4NDQ2OmQ4MWMxZGZiLTFmNGYtNDk5NS05OGQzLTBiMzYyYWJmNjk3OA==' # hakaton token

fast_llm = GigaChat(credentials=GIGA_TOKEN,
               model="GigaChat-Pro-preview",
               scope='GIGACHAT_API_CORP', # нужно для хакатоновского токена
               verify_ssl_certs=False)

long_context_llm = GigaChat(credentials=GIGA_TOKEN,
               model="GigaChat-Pro-preview",
               scope='GIGACHAT_API_CORP', # нужно для хакатоновского токена
               verify_ssl_certs=False)

#  Проверка работы LLM  ================================================================================================
# question = "Что такое AI агент?"
# print('fast_llm')
# print(fast_llm([HumanMessage(content=question)]).content[0:200])
# print('long_context_llm')
# print(long_context_llm([HumanMessage(content=question)]).content[0:200])

#  Создание первоначального конспекта  =================================================================================
direct_gen_outline_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            #"Вы - автор статей для Википедии. Напишите структуру страницы Википедии на заданную пользователем тему. Будьте всесторонними и конкретными.",
            "Вы - автор научных статей. Напишите структуру научной статьи на заданную пользователем тему. Будьте всесторонними и конкретными.",
        ),
        ("user", "{topic}"),
    ]
)
# print(direct_gen_outline_prompt.invoke({"topic": 'test topic'}))

example_topic = "биосовместимые наночастицы"  # Theme
llm_draft = fast_llm(direct_gen_outline_prompt.invoke({"topic": example_topic}).messages)
print('На вывод:\n', llm_draft.content)

#  Развитие тем ========================================================================================================
gen_related_topics_prompt = ChatPromptTemplate.from_template(
    """Я пишу научную статью по упомянутой ниже теме. Пожалуйста, определите и порекомендуйте некоторые страницы Википедии по тесно связанным предметам. Я ищу примеры, которые предоставляют информацию о интересных аспектах, обычно ассоциируемых с этой темой, или примеры, которые помогут мне понять типичное содержание и структуру страниц Википедии для похожих тем.

    Пожалуйста, перечисли несколько дополнительных смежных тем для изучения

    Интересующая тема: {topic}
    """
)


class RelatedSubjects(BaseModel):
    """
    class RelatedSubjects(BaseModel):
        topics: List[str] = Field(
            description="Дополнительные темы для изучения",
        )
    """
    topics: List[str]
    class Config:
        schema_extra = {
            "description": "Дополнительные темы для изучения"
        }


expand_chain = gen_related_topics_prompt | fast_llm.with_structured_output(
    RelatedSubjects
)
# related_subjects = await expand_chain.ainvoke({"topic": example_topic})
related_subjects = expand_chain.ainvoke({"topic": example_topic})
# print(related_subjects)

#  Импортируем файл и создаём базу векторов  ===========================================================================
loader = TextLoader("data/Nano.txt", encoding="utf-8")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
documents = text_splitter.split_documents(documents)
# print(f"Total documents: {len(documents)}")

embeddings = GigaChatEmbeddings(
    credentials="Yjg4MTQzMmUtNDAwMS00NDk0LThjOGUtNmU5ZWQ2YzQ4NDQ2OmQ4MWMxZGZiLTFmNGYtNDk5NS05OGQzLTBiMzYyYWJmNjk3OA==",
    verify_ssl_certs=False,
    # model="GigaChat-Pro-preview",
    scope='GIGACHAT_API_CORP'
)

db = Chroma.from_documents(
    documents,
    embeddings,
    client_settings=Settings(anonymized_telemetry=False),
)

closest_records = db.similarity_search(example_topic, k=5)
# print(closest_records)

qa_chain = RetrievalQA.from_chain_type(fast_llm, retriever=db.as_retriever())
# print(qa_chain(example_topic))

#  Добавляем список литературы  ========================================================================================
#  Вариант 1
# add_referenses_prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             """Вы - автор научных статей. Вы собрали информацию о литературе по этой теме. Теперь вы добавляете список литературы
#             в раздел 'Список литературы'. Вам нужно убедиться, что структура всесторонняя и конкретная.
# Тема, о которой вы пишете: {topic}
#
# Структура статьи:
#
# {old_outline}""",
#         ),
#         (
#             "user",
#             """Добавьте список литературы на основе:\n\nСписок литературы:\n\n{references}\n\nНапишите уточненную структуру статьи
#             с содержанием раздела 'Список литературы':""",
#         ),
#     ]
# )
# print(add_referenses_prompt.invoke({"topic": example_topic, "old_outline": llm_draft, "references" : closest_records[0].page_content}).messages)

# llm_draft_ref=fast_llm(add_referenses_prompt.invoke({"topic": example_topic, "old_outline": llm_draft, "references" : closest_records[0].page_content}).messages)
# print(llm_draft_ref.content)

#  Вариант 3
add_referenses_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Вы - автор научных статей и исследователь.
            Вы собрали список литературы в базу данных.
            Ваша задача на основании темы исследования подобрать ВСЕ научный статьи и прописать в этом формате:
            (Название статьи, автор, ссылка https://cyberleninka.ru/article/n/ugo )
Тема, о которой вы пишете: {topic}

""",
        ),

    ]
)

ref_prompt=add_referenses_prompt.invoke({"topic": example_topic,
                                         "old_outline": llm_draft,
                                         "references" : closest_records[0].page_content}).messages

llm_draft_ref=fast_llm(ref_prompt)
print(llm_draft_ref.content)

#  Генерация точек зрения  =============================================================================================






