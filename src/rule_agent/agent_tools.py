import json
from google.genai import types


# -------------------------------------------------------
# Function Declaration: retrieve_dnd_rules
# -------------------------------------------------------
retrieve_dnd_rules_func = types.FunctionDeclaration(
    name="retrieve_dnd_rules",
    description=(
        "Retrieve Dungeons & Dragons rule passages relevant to a user's intent or action. "
        "Search through the embedded rulebook database for mechanics, conditions, or combat rules."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "query": types.Schema(
                type=types.Type.STRING,
                description="User action or intent (e.g. 'attack the goblin', 'move 30 feet', 'cast fireball')."
            ),
            "n_results": types.Schema(
                type=types.Type.INTEGER,
                description="Number of relevant rule chunks to retrieve.",
                default=5
            ),
        },
        required=["query"]
    ),
)


# -------------------------------------------------------
# Function Implementation
# -------------------------------------------------------
def retrieve_dnd_rules(query, collection, embed_func, n_results=5):
    """
    Search the vector database for the most relevant DnD rules related to the user's query.
    """
    query_embedding = embed_func(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    # Return concatenated rule text
    return "\n\n".join(results["documents"][0])


# -------------------------------------------------------
# Tool Definition
# -------------------------------------------------------
dnd_rule_tool = types.Tool(
    function_declarations=[retrieve_dnd_rules_func]
)


# -------------------------------------------------------
# Execute Function Calls
# -------------------------------------------------------
def execute_function_calls(function_calls, collection, embed_func):
    parts = []
    for function_call in function_calls:
        print("Function:", function_call.name)

        if function_call.name == "retrieve_dnd_rules":
            args = function_call.args
            query = args.get("query")
            n_results = args.get("n_results", 5)

            print(f"Calling retrieve_dnd_rules with query: '{query}'")

            response = retrieve_dnd_rules(query, collection, embed_func, n_results=n_results)
            print("Response:", response[:300], "...")  # print preview only

            parts.append(
                types.Part.from_function_response(
                    name=function_call.name,
                    response={"content": response},
                )
            )

    return parts
