import os
from typing import Type
import fire
import requests
from pathlib import Path
import coloredlogs
from llama_index import (
    GPTListIndex,
    GPTSimpleVectorIndex,
    GPTTreeIndex,
)
from llama_index.readers.schema.base import Document
from llama_index import LLMPredictor, GPTSimpleVectorIndex
from llama_index.indices.base import BaseGPTIndex
from langchain import OpenAI
from langchain.agents import Tool, initialize_agent
from langchain.chains.conversation.memory import ConversationBufferMemory

coloredlogs.install(level="DEBUG")


def main(database_id: str):
    """
    Usage:
        python notionbot.py <notion database id>
    Example:
        python notionbot.py 8405f70a85b44fe7a211a0a56c0d4cc4
        > "Что я делал 22 февраля 2022 года?"
        Ходил в магазин за хлебом
    """
    llm = OpenAI(
        temperature=0, model_name="gpt-3.5-turbo"
    )  # type: ignore  # defauts to text-davinci-003
    index = build_index_for_notion_db(database_id, index_cls=GPTListIndex, llm=llm)
    # index = build_index_for_notion_db(database_id, GPTTreeIndex, llm=llm)

    # Tools that the AI (agent) can use if needed
    when_to_use_the_tool = """
    this tool is useful when you need to answer questions about the author. For example, if I ask you a question about myself, my activities in the past, or other things that only I would know, you can redirect this question to the tool "GPT Index".
    """
    tools = [
        Tool(
            name="GPT Index",
            func=lambda q: str(index.query(q)),
            description=when_to_use_the_tool,  # for AI to understand when to use the tool
            return_direct=True,
        ),
    ]

    memory = ConversationBufferMemory(memory_key="chat_history")
    agent_chain = initialize_agent(
        tools, llm, agent="conversational-react-description", memory=memory
    )

    while True:
        try:
            query = input("Enter query: ")
        except KeyboardInterrupt:
            break
        response = agent_chain.run(query)
        print(response)


NOTION_HEADERS = {
    "Authorization": "Bearer " + os.environ["NOTION_TOKEN"],
    "Notion-Version": "2022-06-28",
    "Content-Type": "application/json",
}


def build_index_for_notion_db(
    database_id: str, index_cls: Type = GPTSimpleVectorIndex, llm=None
) -> BaseGPTIndex:
    """
    Build a Llama-index from a Notion database {database_id}
    """
    save_path = (
        Path(__file__).parent / "saves" / f"{database_id}_{index_cls.__name__}.json"
    )
    if save_path.exists():
        print(f"Loading index from {save_path}")
        return index_cls.load_from_disk(str(save_path))

    print(f"Loading data from the database {database_id}, will write to {save_path}")
    cursor = None
    page_num = 0
    docs = []
    while True:
        page_num += 1
        print(f"Loading page #{page_num}")
        resp = requests.post(
            f"https://api.notion.com/v1/databases/{database_id}/query",
            headers=NOTION_HEADERS,
            json={"start_cursor": cursor} if cursor else {},
        )
        if resp.status_code != 200:
            raise ValueError(
                f"Failed to load data from the database, response: {resp.json()}"
            )
        for page in resp.json()["results"]:
            if (d := page["properties"]["Date"].get("date")) and (
                rt := page["properties"]["Journal"]["rich_text"]
            ):
                journal = rt[0]["text"]["content"]
                docs.append(
                    Document(
                        journal,
                        extra_info={
                            "Дата": d["start"],
                            "Ключевые слова": ", ".join(
                                w["name"]
                                for w in page["properties"]["Key words"]["multi_select"]
                            ),
                            "Тренировки": ", ".join(
                                w["name"]
                                for w in page["properties"]["Workout"]["multi_select"]
                            ),
                        },
                    )
                )
        if not (cursor := resp.json().get("next_cursor")):
            break

    print("Indexing documents")
    index = index_cls(docs, llm_predictor=LLMPredictor(llm) if llm else None)
    index.save_to_disk(str(save_path))
    return index


if __name__ == "__main__":
    fire.Fire(main)
