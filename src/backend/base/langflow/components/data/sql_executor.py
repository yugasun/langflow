from datetime import datetime
import json
from langchain_community.utilities import SQLDatabase
from langchain_community.utilities.sql_database import truncate_word

from langflow.custom import Component
from langflow.inputs.inputs import MultilineInput, StrInput, BoolInput
from langflow.template.field.base import Output
from langflow.schema.message import Message
from loguru import logger


class SQLExecutorComponent(Component):
    display_name = "SQL Query"
    description = "Execute SQL query."
    name = "SQLExecutor"
    beta: bool = True

    inputs = [
        MultilineInput(
            name="query",
            input_types=["Text", "Message"],
            display_name="SQL Query",
            info="The SQL query to execute.",
            placeholder="SELECT * FROM table_name",
        ),
        StrInput(
            name="database_url",
            display_name="Database URL",
            info="The URL of the database.",
            placeholder="postgres://user:password@localhost:5432/dbname",
        ),
        BoolInput(
            name="include_columns",
            display_name="Include Columns",
            value=True,
            info="Include columns in the result.",
            # advanced=True,
        ),
        BoolInput(
            name="passthrough",
            display_name="Passthrough",
            value=False,
            info="If an error occurs, return the query instead of raising an exception.",
            # advanced=True,
        ),
        BoolInput(
            name="add_error",
            display_name="Add Error",
            value=False,
            info="Add the error to the result.",
            # advanced=True,
        ),
    ]

    outputs = [
        Output(display_name="Query Result", name="text", method="run_sql"),
    ]

    def clean_up_uri(self, uri: str) -> str:
        if uri.startswith("postgresql://"):
            uri = uri.replace("postgresql://", "postgres://")
        return uri.strip()

    async def run_sql(self) -> Message:
        query = self.query
        database_url = self.clean_up_uri(self.database_url)
        include_columns = self.include_columns
        passthrough = self.passthrough
        add_error = self.add_error
        error = None
        try:
            database = SQLDatabase.from_uri(database_url)
        except Exception as e:
            msg = f"An error occurred while connecting to the database: {e}"
            raise ValueError(msg) from e
        try:
            cursor = database.run_no_throw(query, fetch="cursor")
            result = [x._asdict() for x in cursor.fetchall()]

            result = [
                {
                    column: truncate_word(value, length=300)
                    for column, value in r.items()
                }
                for r in result
            ]

            # transform the datatime value to timestamp
            if isinstance(result, list):
                for i, row in enumerate(result):
                    for key, value in row.items():
                        if isinstance(value, datetime):
                            result[i][key] = value.timestamp()

            if not include_columns:
                result = [tuple(row.values()) for row in result]

            result = json.dumps(result)

            self.status = result
        except Exception as e:
            result = str(e)
            self.status = result
            if not passthrough:
                raise
            error = repr(e)

        if add_error and error is not None:
            result = f"{result}\n\nError: {error}\n\nQuery: {query}"
        elif error is not None:
            # Then we won't add the error to the result
            # but since we are in passthrough mode, we will return the query
            result = query

        logger.debug(f"SQLExecutorComponent: {result}")
        logger.debug(f"Result type: {type(result)}")

        return Message(
            text=result,
        )
