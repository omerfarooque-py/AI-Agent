def convert_chat_history(history_dicts):
    chat_history_tuples = []
    for i in range(0, len(history_dicts), 2):
        user_msg = history_dicts[i]["content"]
        assistant_msg = ""
        if i+1 < len(history_dicts):
            assistant_msg = history_dicts[i+1]["content"]
        chat_history_tuples.append((user_msg, assistant_msg))
    return chat_history_tuples