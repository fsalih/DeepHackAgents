from langchain.chat_models.gigachat import GigaChat
# from langchain.schema import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Optional
from langchain.document_loaders import TextLoader
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
)
from chromadb.config import Settings
from langchain.vectorstores import Chroma
from langchain_community.embeddings import GigaChatEmbeddings


GIGA_TOKEN='Yjg4MTQzMmUtNDAwMS00NDk0LThjOGUtNmU5ZWQ2YzQ4NDQ2OmQ4MWMxZGZiLTFmNGYtNDk5NS05OGQzLTBiMzYyYWJmNjk3OA==' # hakaton token


# Инициируем LLM
fast_llm = GigaChat(credentials=GIGA_TOKEN,
               model="GigaChat-Pro-preview",
               scope='GIGACHAT_API_CORP', # нужно для хакатоновского токена
               verify_ssl_certs=False)

long_context_llm = GigaChat(credentials=GIGA_TOKEN,
               model="GigaChat-Pro-preview",
               scope='GIGACHAT_API_CORP', # нужно для хакатоновского токена
               verify_ssl_certs=False)


def agent(message):
    example_topic = message
    # Проверка работы LLM
    # question = "Что такое AI агент?"
    # print('fast_llm')
    # print(fast_llm([HumanMessage(content=question)]).content[0:200])
    # print('long_context_llm')
    # print(long_context_llm([HumanMessage(content=question)]).content[0:200])

    # Создание первоначального конспекта
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

    llm_draft=fast_llm(direct_gen_outline_prompt.invoke({"topic": example_topic}).messages)
    doc_structure = llm_draft.content
    # print(llm_draft.content)

    # Развитие тем
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

    related_subjects = expand_chain.ainvoke({"topic": example_topic})
    # print(related_subjects)

    # Импортируем файл и создаём базу векторов
    loader = TextLoader("data/Nano.txt", encoding="utf-8")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    documents = text_splitter.split_documents(documents)
    # print(f"Total documents: {len(documents)}")

    embeddings = GigaChatEmbeddings(
        credentials=GIGA_TOKEN,
        verify_ssl_certs=False,

        # model="GigaChat-Pro-preview",
        scope='GIGACHAT_API_CORP'
    )

    db = Chroma.from_documents(
        documents,
        embeddings,
        client_settings=Settings(anonymized_telemetry=False),
    )

    # Запрашиваем из векторной БД наиболее близкие записи
    closest_records = db.similarity_search(example_topic, k=10)
    # print(closest_records)

    # Добавляем список литературы
    # Вариант 3 (хорошо)
    add_referenses_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Вы - автор научных статей и исследователь.
                Вы собрали список литературы в базу данных.
                Ваша задача на основании темы исследования подобрать ВСЕ научный статьи и прописать в этом формате:
                (Название статьи, автор, ссылка https://cyberleninka.ru/article/n/ugo )
    
    Список литературы: {references}
    """,
            ),

        ]
    )

    ref_prompt=add_referenses_prompt.invoke({"references" : closest_records}).messages
    llm_draft_ref=fast_llm(ref_prompt)
    literature_list = llm_draft_ref.content

    # "биосовместимые наночастицы"
    # print(f'Структура документа: {doc_structure} \n\n\n{literature_list}')
    return(f'Структура документа: {doc_structure} \n\n\n{literature_list}')







