# agents/llama3_agent.py

import subprocess
import os
import re
from typing import Optional, List, Dict, Generator

from core.ollama_client import OllamaClient

class Llama3Agent:
    def __init__(self):
        self.ollama_client = OllamaClient()
        self.model_name = "llama3.2:1b-instruct-q2_K"
        self.embedding_model_name = "nomic-embed-text"

        self.system_message = (
            "You are an intelligent OS assistant integrated into a custom operating system. "
            "Your core purpose is to help the user interact with and manage their system, "
            "as well as answer general questions concisely. "
            "Always be aware of the 'Current Working Directory'. "
            "When the user asks you to perform a system action (like listing files, creating folders, running programs, etc.), "
            "you must respond in one of the following formats:\n\n"
            "1.  **For Shell Commands:** If a shell command is needed, output ONLY the command prefixed with `[COMMAND]:` on a new line, like this: `[COMMAND]:ls -l`\n"
            "    * Ensure the command is part of the whitelisted commands. "
            "    * Handle `cd` commands internally by updating your current directory. "
            "    * Do NOT include any other text (like 'Okay, here's the command:') when giving a command.\n"
            "2.  **For Python Code:** If the request requires Python code, output the code in a standard markdown code block (```python\\n...\\n```).\n"
            "3.  **For General Answers:** For conversational responses or information that doesn't involve system actions, respond directly and concisely.\n\n"
            "Prioritize system interactions when clearly requested. Be direct and avoid unnecessary conversational filler."
        )
        self.conversation_history: List[Dict[str, str]] = []

        self.whitelisted_commands = {
            "ls": True, "pwd": True, "cat": True, "echo": True, "find": True,
            "grep": True, "df": True, "du": True, "ps": True, "whoami": True,
            "mkdir": True, "rmdir": True, "touch": True, "rm": True,
        }
        self.current_working_directory = os.path.expanduser("~")

    def _execute_shell_command(self, command_string: str) -> str:
        cmd_parts = command_string.strip().split(maxsplit=1)
        if not cmd_parts:
            return "Error: No command provided for execution."

        base_command = cmd_parts[0]

        if ".." in command_string.split('/') and base_command not in ["cd", "ls", "cat", "find", "grep"]:
             return "Security Alert: Command contains '..' and is not a permitted navigation/read command. Execution blocked."

        if base_command not in self.whitelisted_commands or not self.whitelisted_commands[base_command]:
            return f"Error: Command '{base_command}' is not explicitly whitelisted for execution."

        print(f"[Agent] Executing command (safely): '{command_string}' in '{self.current_working_directory}'")
        try:
            result = subprocess.run(
                command_string,
                shell=True,
                capture_output=True,
                text=True,
                check=True,
                cwd=self.current_working_directory
            )
            output = result.stdout.strip()
            error_output = result.stderr.strip()

            if output:
                return f"Command Output:\n```bash\n{output}\n```"
            elif error_output:
                return f"Command Error:\n```bash\n{error_output}\n```"
            else:
                return "Command executed successfully with no output."

        except subprocess.CalledProcessError as e:
            return f"Command failed with exit code {e.returncode}:\nError Output:\n```bash\n{e.stderr.strip()}\n```"
        except FileNotFoundError:
            return f"Error: Command '{base_command}' not found. Is it installed?"
        except Exception as e:
            return f"An unexpected error occurred during command execution: {e}"

    def process_query(self, query: str) -> str:
        dynamic_system_message = self.system_message + \
            f"\n\nCurrent Working Directory: {self.current_working_directory}"

        self.conversation_history.append({"role": "user", "content": query})
        print(f"[Agent] Processing query: '{query}'...")

        messages_for_ollama = [{"role": "system", "content": dynamic_system_message}]
        messages_for_ollama.extend(self.conversation_history)

        response_generator = self.ollama_client.generate_text(
            model_name=self.model_name,
            messages=messages_for_ollama,
            stream=True,
            temperature=0.1,
            max_tokens=500
        )

        if response_generator is None:
            print("[Agent] Error: Could not get a streaming response from Ollama client.")
            return "I'm sorry, I'm having trouble connecting to the model right now. Please try again later."

        full_response = ""
        for chunk in response_generator:
            if chunk:
                full_response += chunk

        final_response = full_response.strip()
        print(f"[Agent] LLM Generated Raw Response: '{final_response}'")

        command_match = re.match(r'\[COMMAND\]:\s*(.*)', final_response, re.DOTALL)

        if command_match:
            command_string = command_match.group(1).strip()
            print(f"[Agent] Detected structured command: '{command_string}'.")

            if command_string.startswith("cd "):
                new_path = command_string[3:].strip()
                absolute_new_path = os.path.abspath(os.path.join(self.current_working_directory, new_path))
                
                if os.path.exists(absolute_new_path) and os.path.isdir(absolute_new_path):
                    self.current_working_directory = absolute_new_path
                    response_msg = f"Changed current working directory to: `{self.current_working_directory}`"
                else:
                    response_msg = f"Error: Cannot change directory to `{new_path}`. Path does not exist or is not a directory."
                
                print(f"[Agent] Internal CD handled: {response_msg}")
                self.conversation_history.append({"role": "assistant", "content": response_msg})
                return response_msg
            else:
                execution_result = self._execute_shell_command(command_string)
                self.conversation_history.append({"role": "assistant", "content": execution_result})
                return execution_result

        elif final_response.startswith("```python"):
            print("[Agent] Detected Python code block.")
            self.conversation_history.append({"role": "assistant", "content": final_response})
            return final_response

        else:
            print("[Agent] Detected general conversational response.")
            self.conversation_history.append({"role": "assistant", "content": final_response})
            return final_response

    def get_embeddings_for_text(self, text: str) -> Optional[List[float]]:
        print(f"[Agent] Generating embeddings for text (using {self.embedding_model_name})...")
        embeddings = self.ollama_client.get_embeddings(self.embedding_model_name, text)
        if embeddings is None:
            print("[Agent] Error: Could not generate embeddings.")
        return embeddings
