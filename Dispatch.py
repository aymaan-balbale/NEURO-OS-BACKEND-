# dispatcher.py

from agents.llama3_agent import Llama3Agent

class Dispatcher:
    def __init__(self):
        self.llama3_agent = Llama3Agent()
        print("{Dispatcher} Llama3 Agent initialized and ready to handle all tasks.")

    def dispatch_request(self, user_query: str) -> str:
        print(f"\n{{Dispatcher}} Received query: '{user_query}'")
        
        response_content = self.llama3_agent.process_query(user_query)
        
        print(f"{{Dispatcher}} Final response from Llama3 Agent: '{response_content[:75]}...'")
        return response_content
