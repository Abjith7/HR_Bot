from collections import deque

class ConversationMemory:
    def __init__(self, max_turns=6):
        self.history = deque(maxlen=max_turns)
        self.active_documents = []

    def add_turn(self, user, assistant, sources):
        self.history.append({
            "user": user,
            "assistant": assistant,
            "sources": sources
        })
        if sources:
            self.active_documents = sources

    def get_context(self):
        context = ""
        for turn in self.history:
            context += f"User: {turn['user']}\n"
            context += f"Assistant: {turn['assistant']}\n\n"
        return context.strip()

    def get_active_documents(self):
        return self.active_documents
