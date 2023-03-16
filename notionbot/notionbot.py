import os
from typing import Type
import requests
from pathlib import Path
import re

import fire
import coloredlogs
import pandas as pd
import sqlite3
from llama_index import (
    GPTSQLStructStoreIndex,
    LLMPredictor,
    Prompt,
    SQLDatabase,
)
from llama_index.readers.schema.base import Document
from llama_index.indices.base import BaseGPTIndex
from langchain import OpenAI
from langchain.llms.base import BaseLLM
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent import AgentExecutor
from langchain.chains.conversation.memory import ConversationBufferMemory
from sqlalchemy import create_engine

coloredlogs.install(level="DEBUG")


def main(csv_path: str):
    """
    Argument: a path to a CSV file with required entries "Date" and "Journal", e.g. exported from a Notion database.

    It will build a corresponding SQL database and a correspondng JSON llama-index, whichyou can query with the following questions:

    > "Что я делал 22 февраля 2022 года?"
    Играл в теннис.

    > "Сколько раз я играл в теннис в 2022 году?"
    22 раза.

    > "В какие дни?"
    Apr 1, Apr 17, May 4, ...
    """
    # Use the turbo ChatGPT which is 10x cheeper.
    llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo")  # type: ignore

    # Build a unstructured+structured Llama index.
    index = build_index(Path(csv_path), LLMPredictor(llm))

    # Build a Langchain agent that will use our index as a tool.
    agent = build_agent(index, llm)

    while True:
        try:
            query = input("Enter query: ")
        except KeyboardInterrupt:
            break
        response = agent.run(query)
        print(response)


def build_index(csv_path: Path, llm_predictor: LLMPredictor) -> BaseGPTIndex:
    """
    Build a Llama-index from a CSV database.

    Llama index provides multiple ways to index the data.

    * GPTSimpleVectorIndex only gives AI the most top K similar documents as determined from embeddings, exluding everything else from the context. So there is no way to answer statistical questions.

    * GPTListIndex iterates over each page (with optional filters), making an expensive ChatGPT query for each journal entry.

    * GPTTreeIndex recursively summarizes groups of journal entries, essentially removing statistics that could be collected from daily entries.

    * GPTSQLStructStoreIndex both takes unstrucutred data and structured SQL entries, which like what we need for a table-like data. It still would make an expensive query per each entry, however, we can aggregate them per-month.
    It still should fit the context length, and keep the total number of documents, and this ChatGPT queries, low.
    """
    TABLE_NAME = "days"

    # Will write two intermediate files:
    db_path = csv_path.with_suffix(".db")
    index_path = csv_path.with_suffix(".json")
    if db_path.exists() and index_path.exists():
        print(f"Loading existing index from disk {index_path} (db: {db_path})")
        return GPTSQLStructStoreIndex.load_from_disk(
            str(index_path),
            sql_database=SQLDatabase(create_engine(f"sqlite:///{db_path}")),
            table_name=TABLE_NAME,
            llm_predictor=llm_predictor,
        )

    # Pre-process the dataframe
    df = pd.read_csv(csv_path)

    df = df.dropna(subset=["Date", "Journal"])

    # Removing [OLD] and [MOVED] columns
    df = df.loc[:, ~df.columns.str.contains(r"\[")]

    # Fix column names with special characters:
    df.columns = df.columns.str.replace(r"\s", "_", regex=True)

    # Fix relation fields (they are exported as URLs instead of titles)
    def _fix_relation_fields(x):
        if isinstance(x, str):
            vals = []
            for v in x.split(", "):
                if v.startswith("https://www.notion.so/"):
                    v = re.sub(r"https://www.notion.so/([\w-]+)-\w+", r"\1", v)
                    v = v.replace("-", " ")
                    if v.startswith("https://www.notion.so/"):
                        continue
                vals.append(v)
            x = "; ".join(vals)
        return x

    df = df.applymap(_fix_relation_fields)

    # Save the CSV to a database
    conn = sqlite3.connect(db_path)
    df.to_sql(TABLE_NAME, conn, if_exists="replace", index=True)
    conn.close()

    # Getting our unstructured documents for an index. We don't want a document
    # for each entry, as it would lead to a lot of ChatGPT queries. So we aggregate
    # document per month. That shouldn't exceed the ChatGPT context length, and
    # should keep the number of documetns limited.
    df["MonthYear"] = pd.to_datetime(df["Date"]).dt.to_period("M")
    df["Journal"] = df["Date"] + "\n\n" + df["Journal"]
    monthly = df.groupby("MonthYear").agg({"Journal": ";".join})

    docs = [Document(journal) for journal in monthly["Journal"]]

    index = GPTSQLStructStoreIndex(
        docs,
        sql_database=SQLDatabase(create_engine(f"sqlite:///{db_path}")),
        table_name=TABLE_NAME,
        llm_predictor=llm_predictor,
    )
    index.save_to_disk(str(index_path))
    return index


def build_agent(index: BaseGPTIndex, llm: BaseLLM) -> AgentExecutor:
    """
    Build a Langchain agent that can can use index as a tool.
    """
    tool = Tool(
        name="GPT Index",
        func=lambda q: str(index.query(q)),
        # For AI to understand when to use the tool:
        description="""
        this tool is useful when you need to answer questions about the author. For example, if I ask you a question about myself, my activities in the past, or other things that only I would know, you can redirect this question to the tool "GPT Index".
        """,
        return_direct=True,
    )
    return initialize_agent(
        tools=[tool],
        llm=llm,
        agent="conversational-react-description",
        memory=ConversationBufferMemory(memory_key="chat_history"),
    )


if __name__ == "__main__":
    fire.Fire(main)
